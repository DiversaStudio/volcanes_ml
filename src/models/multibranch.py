import torch
import torch.nn as nn

class MultiBranchModel(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.6):
        super(MultiBranchModel, self).__init__()
        
        # Print input dimensions for verification
        print("Expected input dimensions:")
        print("Corrected & Edge branches: [batch_size, 1, 240, 320]")
        print("Threshold branch: [batch_size, 4, 240, 320]")
        
        # Branch for corrected thermal images
        self.corrected_branch = nn.Sequential(
            # Reduce spatial dimensions gradually
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Output: [16, 240, 320]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2, 2),                                    # Output: [16, 120, 160]
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output: [32, 60, 80]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                    # Output: [32, 30, 40]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output: [64, 15, 20]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                         # Output: [64, 1, 1]
            nn.Flatten()                                          # Output: [64]
        )
        
        # Branch for edge detection (same architecture)
        self.edge_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Branch for threshold tensors
        self.threshold_branch = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Fully connected layers
        # Each branch outputs 64 features, so total is 64 * 3 = 192
        total_features = 64 * 3
        
        self.fc = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, n_classes)
        )

    def forward(self, corrected, edge, thresholds):
        # Process each branch
        x1 = self.corrected_branch(corrected)     # Output: [batch_size, 64]
        x2 = self.edge_branch(edge)               # Output: [batch_size, 64]
        
        # Use the thresholds tensor directly
        x3 = self.threshold_branch(thresholds)    # Output: [batch_size, 64]
        
        # Concatenate all features
        x = torch.cat((x1, x2, x3), dim=1)        # Output: [batch_size, 192]
        
        # Final classification
        x = self.fc(x)                            # Output: [batch_size, n_classes]
        return x