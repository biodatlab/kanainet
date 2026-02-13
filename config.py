"""
Configuration file for KAN-ACNet training
"""

class Config:
    """
    Training configuration
    """
    # Data
    data_root = './data/kvasir-seg'  # Path to Kvasir-SEG dataset
    image_size = (512, 512)
    val_split = 0.2
    batch_size = 2
    num_workers = 4

    # Model
    model_name = 'KAN-ACNet'
    in_channels = 3
    num_classes = 1  # Binary segmentation
    base_channels = 64
    use_pretrained_backbone = True
    use_texture_pathway = True
    backbone = 'convnext_tiny'  # Options: 'convnext_tiny', 'resnet34', 'resnet50', 'efficientnet_v2_s'

    # Training
    num_epochs = 100
    learning_rate = 1e-4
    weight_decay = 1e-5
    optimizer = 'adamw'  # 'adam', 'adamw', 'sgd'
    scheduler = 'cosine'  # 'cosine', 'step', 'plateau', None
    early_stopping_patience = 20

    # Loss weights
    bce_weight = 1.0
    dice_weight = 1.0
    iou_weight = 0.5
    focal_weight = 0.5
    boundary_weight = 0.5
    use_deep_supervision = True

    # Device
    device = 'cuda'  # 'cuda' or 'cpu'
    seed = 42

    # Checkpoints and logging
    checkpoint_dir = './checkpoints'
    log_dir = './logs'
    save_frequency = 25  # Save checkpoint every N epochs

    # Resume training
    resume = False
    resume_checkpoint = './checkpoints/best_model.pth'

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }

    @classmethod
    def print_config(cls):
        """Print configuration"""
        print("=" * 60)
        print("Configuration:")
        print("=" * 60)
        for key, value in cls.to_dict().items():
            print(f"{key:30s}: {value}")
        print("=" * 60)
