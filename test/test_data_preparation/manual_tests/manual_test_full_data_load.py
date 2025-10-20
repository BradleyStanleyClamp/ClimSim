
from omegaconf import OmegaConf
import data_preparation

def manual_test_full_data_load_sub_sampled_low_res(testing_type: str = 'full'):
    """
    Manual test to load the full sub_sampled_low_res dataset and print its shape.
    """

    # Load configuration
    dataset_cfg = OmegaConf.load('config/dataset/sub_sampled_low_res.yaml')
    

    # Load full dataset
    print("Loading SubSampledLowResDataset with testing type:", testing_type)
    trainset, valset, testset = data_preparation.get_all_datasets(dataset_cfg, dataset_testing_type=testing_type)
  

    print(f"Trainset loaded with {len(trainset)} samples.")
    print(f"Valset loaded with {len(valset)} samples.")
    print(f"Testset loaded with {len(testset)} samples.")


if __name__ == "__main__":
    manual_test_full_data_load_sub_sampled_low_res()
