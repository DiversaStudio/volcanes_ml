import os
from datetime import datetime
from tqdm import tqdm
from flirpy.io.fff import Fff
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

class ThermalDataLoader:
    """Handles loading and basic processing of FLIR FFF thermal image files."""
    
    def __init__(self, base_directories, allowed_labels=None, limit_per_directory=None):
        self.base_directories = base_directories
        self.allowed_labels = allowed_labels
        self.limit_per_directory = limit_per_directory
        self.dataset_summary = {
            'processed_files': {},
            'skipped_files': {},
            'date_ranges': {}
        }

    def process_fff_files(self, directory, label):
        """Process multiple FFF thermal image files from a directory."""
        data = []
        problematic_files = []

        # Get list of FFF files
        fff_files = [f for f in os.listdir(directory) if f.endswith('.fff')]
        if self.limit_per_directory:
            fff_files = fff_files[:self.limit_per_directory]

        # Process files with progress bar
        for filename in tqdm(fff_files, desc=f"Processing files in {label}"):
            file_path = os.path.join(directory, filename)
            try:
                # Process individual image
                raw_image, corrected_image, metadata, raw_global_stats, \
                corrected_global_stats, raw_segment_stats, corrected_segment_stats = \
                    self.process_thermal_image(file_path)

                # Extract timestamp
                timestamp = self._extract_timestamp(metadata, filename)

                # Store processed data
                data.append((
                    timestamp, raw_image, corrected_image, filename, label,
                    raw_global_stats, corrected_global_stats, raw_segment_stats,
                    corrected_segment_stats
                ))

            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                problematic_files.append((filename, str(e)))

        if problematic_files:
            print("\nProblematic files summary:")
            for filename, error in problematic_files:
                print(f"  {filename}: {error}")

        return sorted(data, key=lambda x: x[0])

    def process_thermal_image(self, file_path):
        """Process a single thermal image from an FFF file."""
        # Load FFF file
        fff_reader = Fff(file_path)
        raw_image = fff_reader.get_image()
        metadata = fff_reader.meta
        metadata['file_path'] = file_path

        # Apply radiometric corrections
        thermal_image_corrected = self._apply_radiometric_corrections(raw_image, metadata)

        # Get temperature range from metadata
        min_temp = float(metadata['CameraTemperatureRangeMin'])
        max_temp = float(metadata['CameraTemperatureRangeMax'])
        thermal_image_clipped = np.clip(thermal_image_corrected, min_temp, max_temp)

        # Calculate statistics
        from src.features.zonal import calculate_zonal_statistics, segment_image
        raw_global_stats = calculate_zonal_statistics(raw_image)
        corrected_global_stats = calculate_zonal_statistics(thermal_image_clipped)

        raw_segments = segment_image(raw_image)
        corrected_segments = segment_image(thermal_image_clipped)
        raw_segment_stats = [calculate_zonal_statistics(segment) for segment in raw_segments]
        corrected_segment_stats = [calculate_zonal_statistics(segment) for segment in corrected_segments]

        return (raw_image, thermal_image_clipped, metadata, raw_global_stats,
                corrected_global_stats, raw_segment_stats, corrected_segment_stats)

    def _extract_timestamp(self, metadata, filename):
        """Extract timestamp from metadata."""
        timestamp_str = metadata.get('Datetime (UTC)', '')
        if timestamp_str:
            try:
                return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise ValueError(f"Could not parse timestamp '{timestamp_str}' for file {filename}")
        
        timestamp_unix = metadata.get('Timestamp')
        if timestamp_unix is None:
            raise ValueError(f"No timestamp information found for file {filename}")
        return datetime.utcfromtimestamp(timestamp_unix)

    def _apply_radiometric_corrections(self, raw_counts, metadata):
        """Apply radiometric corrections to raw sensor counts."""
        # Extract metadata
        R1 = metadata['Planck R1']
        R2 = metadata['Planck R2']
        B = metadata['Planck B']
        F = metadata['Planck F']
        O = metadata['Planck O']
        emissivity = metadata['Emissivity']
        
        # Calculate atmospheric transmission
        tau = self._calculate_atmospheric_transmission(metadata)
        
        # Convert raw to radiance
        raw_refl = R1 / (R2 * (np.exp(B / metadata['Reflected Apparent Temperature']) - F)) - O
        raw_atm = R1 / (R2 * (np.exp(B / metadata['Atmospheric Temperature']) - F)) - O
        
        # Apply corrections
        raw_obj = (raw_counts - (1 - emissivity) * tau * raw_refl -
                  (1 - tau) * raw_atm) / (emissivity * tau)
        
        # Convert to temperature
        return B / np.log(R1 / (R2 * (raw_obj + O)) + F) - 273.15

    def _calculate_atmospheric_transmission(self, metadata):
        """Calculate atmospheric transmission coefficient."""
        X = metadata['Atmospheric Trans X']
        alpha1 = metadata['Atmospheric Trans Alpha 1']
        alpha2 = metadata['Atmospheric Trans Alpha 2']
        beta1 = metadata['Atmospheric Trans Beta 1']
        beta2 = metadata['Atmospheric Trans Beta 2']
        distance = metadata['Object Distance']
        humidity = metadata['Relative Humidity']
        
        return X * np.exp(-np.sqrt(distance) * (alpha1 + beta1 * np.sqrt(humidity))) + \
               (1 - X) * np.exp(-np.sqrt(distance) * (alpha2 + beta2 * np.sqrt(humidity)))

    def create_dataset(self):
        """Create a complete dataset from all directories."""
        all_data = []

        # Process each directory
        for directory in self.base_directories:
            try:
                # Extract and validate label
                label = os.path.basename(directory)
                if self.allowed_labels and label not in self.allowed_labels:
                    print(f"Skipping directory {directory} - label {label} not in allowed labels")
                    continue

                # Process directory
                print(f"\nProcessing directory: {directory}")
                sorted_data = self.process_fff_files(directory, label)

                # Update summary
                self.dataset_summary['processed_files'][label] = len(sorted_data)
                if sorted_data:
                    self.dataset_summary['date_ranges'][label] = {
                        'start': sorted_data[0][0],
                        'end': sorted_data[-1][0]
                    }
                    all_data.extend(sorted_data)
                    
            except Exception as e:
                print(f"Error processing directory {directory}: {str(e)}")
                self.dataset_summary['skipped_files'][label] = str(e)

        # Sort and remove duplicates
        print("\nFinalizing dataset...")
        all_data.sort(key=lambda x: x[0])
        return self.create_tensors(all_data)

    def create_tensors(self, sorted_data):
        """Create tensors from processed data."""
        if not sorted_data:
            raise ValueError("No data to process")

        # Get dimensions from first image
        height, width = sorted_data[0][1].shape
        
        # Initialize tensors and lists
        raw_tensor = np.zeros((height, width, len(sorted_data)))
        corrected_tensor = np.zeros((height, width, len(sorted_data)))
        timestamps = []
        filenames = []
        labels = []
        raw_global_stats = []
        corrected_global_stats = []
        raw_segment_stats = []
        corrected_segment_stats = []

        # Fill tensors and lists
        for i, data in enumerate(sorted_data):
            timestamps.append(data[0])
            raw_tensor[:,:,i] = data[1]
            corrected_tensor[:,:,i] = data[2]
            filenames.append(data[3])
            labels.append(data[4])
            raw_global_stats.append(data[5])
            corrected_global_stats.append(data[6])
            raw_segment_stats.append(data[7])
            corrected_segment_stats.append(data[8])

        return {
            'tensors': {
                'raw': raw_tensor,
                'corrected': corrected_tensor
            },
            'metadata': {
                'timestamps': timestamps,
                'filenames': filenames,
                'labels': labels
            },
            'statistics': {
                'raw_global': raw_global_stats,
                'corrected_global': corrected_global_stats,
                'raw_segments': raw_segment_stats,
                'corrected_segments': corrected_segment_stats
            },
            'summary': {
                'total_images': len(sorted_data),
                'tensor_shape': raw_tensor.shape,
                'unique_labels': list(set(labels)),
                **self.dataset_summary
            }
        }