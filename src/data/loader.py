import os
from datetime import datetime
from tqdm import tqdm
from flirpy.io.fff import Fff
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from src.features.zonal import calculate_zonal_statistics

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
 
    def process_thermal_image(self, file_path):
        """Process a single thermal image."""
        try:
            # Load FFF file
            fff_reader = Fff(file_path)
            raw_image = fff_reader.get_image()
            
            # Check if raw image is valid
            if raw_image is None or np.isnan(raw_image).all():
                print(f"Skipping {file_path}: Invalid or all NaN values in raw image")
                return None
            
            # Extract metadata
            metadata = fff_reader.meta
            metadata['file_path'] = file_path

            # Apply radiometric corrections
            thermal_image_corrected = self._apply_radiometric_corrections(raw_image, metadata)
            
            # Check if corrected image has all NaN values
            if np.isnan(thermal_image_corrected).all():
                print(f"Skipping {file_path}: All NaN values after correction")
                return None

            # Continue processing only if image is valid
            min_temp = float(metadata['CameraTemperatureRangeMin'])
            max_temp = float(metadata['CameraTemperatureRangeMax'])
            thermal_image_clipped = np.clip(thermal_image_corrected, min_temp, max_temp)
            
            # Calculate statistics
            raw_global_stats = calculate_zonal_statistics(raw_image)
            corrected_global_stats = calculate_zonal_statistics(thermal_image_clipped)
        
            return (raw_image, thermal_image_clipped, metadata, raw_global_stats, 
                    corrected_global_stats)
                    
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    def process_fff_files(self, directory, label):
        """Process multiple FFF files from a directory."""
        data = []
        problematic_files = []

        # Get list of FFF files
        fff_files = [f for f in os.listdir(directory) if f.endswith('.fff')]
        if self.limit_per_directory is not None:
            fff_files = fff_files[:self.limit_per_directory]

        # Process files with progress bar
        for filename in tqdm(fff_files, desc=f"Processing files in {label}"):
            file_path = os.path.join(directory, filename)
            try:
                # Process individual image
                result = self.process_thermal_image(file_path)
                
                if result is None:
                    problematic_files.append((filename, "Invalid or NaN image - skipped"))
                    continue
                    
                raw_image, corrected_image, metadata, raw_global_stats, corrected_global_stats = result

                # Extract timestamp
                try:
                    timestamp = self._extract_timestamp(metadata, filename)
                except ValueError as e:
                    problematic_files.append((filename, str(e)))
                    continue

                # Store processed data
                data.append((
                    timestamp, raw_image, corrected_image, filename, label,
                    raw_global_stats, corrected_global_stats
                ))

            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                problematic_files.append((filename, str(e)))

        # Report problematic files
        if problematic_files:
            print("\nProblematic files summary:")
            for filename, error in problematic_files:
                print(f"  {filename}: {error}")
            print(f"\nTotal skipped files: {len(problematic_files)}")

        print(f"\nSuccessfully processed {len(data)} files from {label}")
        
        # Return sorted data by timestamp
        return sorted(data, key=lambda x: x[0])

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
                # Get absolute path
                directory = os.path.abspath(directory)
                
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
        if not all_data:
            return None
            
        all_data.sort(key=lambda x: x[0])
        return self.create_tensors(all_data)

    def create_tensors(self, sorted_data):
        """Create tensors from processed data with memory optimization."""
        if not sorted_data:
            raise ValueError("No data to process")

        # Get dimensions from first image
        height, width = sorted_data[0][1].shape
        n_samples = len(sorted_data)
        
        print(f"\nCreating tensors for {n_samples} images...")
        print(f"Image dimensions: {height}x{width}")
        
        # Calculate memory requirements
        sample_size = height * width * 4  # 4 bytes for float32
        total_size_gb = (sample_size * n_samples * 2) / (1024**3)  # *2 for raw and corrected
        print(f"Estimated memory requirement: {total_size_gb:.2f} GB")

        try:
            # Initialize tensors with float32
            print("\nAllocating memory...")
            raw_tensor = np.zeros((height, width, n_samples), dtype=np.float32)
            corrected_tensor = np.zeros((height, width, n_samples), dtype=np.float32)
            
            # Initialize lists
            timestamps = []
            filenames = []
            labels = []
            raw_global_stats = []
            corrected_global_stats = []

            # Process in batches
            batch_size = 500  # Adjust based on your memory
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                print(f"\nProcessing batch {i//batch_size + 1}/{(n_samples + batch_size - 1)//batch_size}")
                
                # Process batch
                batch_data = sorted_data[i:end_idx]
                for j, data in enumerate(batch_data):
                    idx = i + j
                    timestamps.append(data[0])
                    raw_tensor[:,:,idx] = data[1]
                    corrected_tensor[:,:,idx] = data[2]
                    filenames.append(data[3])
                    labels.append(data[4])
                    raw_global_stats.append(data[5])
                    corrected_global_stats.append(data[6])

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
                    'corrected_global': corrected_global_stats
                },
                'summary': {
                    'total_images': n_samples,
                    'tensor_shape': raw_tensor.shape,
                    'unique_labels': list(set(labels)),
                    **self.dataset_summary
                }
            }
        
        except MemoryError as e:
            print(f"\nMemory Error: Unable to allocate required memory.")
            print("Consider:")
            print("1. Reducing the number of images")
            print("2. Processing in smaller batches")
            print("3. Using less precision (e.g., float16)")
            raise e

        except Exception as e:
            print(f"\nError creating tensors: {str(e)}")
            raise e
