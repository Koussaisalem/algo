import torch
from torch_geometric.data import Data, DataLoader
from models.surrogate import Surrogate, SimpleSurrogate

def test_model(model_class, model_name):
    print(f"--- Testing {model_name} ---")
    try:
        # 1. Instantiate the model
        model = model_class()
        print("✓ Model instantiated successfully")

        # 2. Create sample molecular data
        z = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long) # Methane (CH4)
        pos = torch.randn(5, 3)
        sample_data = Data(z=z, pos=pos)
        print("✓ Sample data created")

        # 3. Perform a forward pass
        output = model(sample_data)
        print(f"✓ Forward pass successful. Output: {output}")

        # 4. Verify output shape
        assert output.shape == (1,), f"Expected output shape (1,), but got {output.shape}"
        print("✓ Output shape is correct")

        # 5. Test with DataLoader and batched data
        data_list = [sample_data, sample_data]
        loader = DataLoader(data_list, batch_size=2)
        batch = next(iter(loader))
        batch_output = model(batch)
        assert batch_output.shape == (2,), f"Expected batch output shape (2,), but got {batch_output.shape}"
        print("✓ Batched forward pass successful")

        print(f"--- {model_name} Test Passed ---")
        return True
    except Exception as e:
        print(f"✗ {model_name} Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model(Surrogate, "Custom SchNet Surrogate")
    print("\n")
    test_model(SimpleSurrogate, "Simple GCN Fallback")