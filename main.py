import yaml
from utils.data_generator import DatasetGenerator

def load_config(config_path):
    """Load configuration settings from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# The entry point of the script
if __name__ == "__main__":
    # Load configuration
    config = load_config("config.yaml")

    # Initialize DatasetGenerator with config settings
    data_loader = DatasetGenerator(
        dataset_name=config['dataset_name'],
        root=config['root'],
        batch_size=config['batch_size'],
        subset_size=config['subset_size'],
        image_size=tuple(config['image_size'])
    )

    # Display data summary
    data_loader.datasummary()

    # Retrieve DataLoaders for train and test sets
    train_loader, test_loader = data_loader.get_loaders()

    # Example of iterating through the train loader
    print("\nIterating through train_loader:")
    for images, labels in train_loader:
        print("Batch of images:", images.shape)
        print("Batch of labels:", labels)
        break  # Print only the first batch
