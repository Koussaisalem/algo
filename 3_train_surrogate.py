import torch
torch.set_default_dtype(torch.float32)
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import os
import traceback
import sys
from models.surrogate import model

# --- Configuration ---
# Keep float32 as the default for library compatibility
torch.set_default_dtype(torch.float32) 

# Use float64 for your model's specific needs
DTYPE = torch.float64 
DEVICE = torch.device('cuda')

DATA_PATH = 'data/qm9_micro_enriched.pt'
MODEL_SAVE_PATH = 'models/surrogate_frozen.pt'

BATCH_SIZE = 10  # Using a smaller batch size for the toy dataset
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

def validate_data_entry(entry, idx):
    """Validate a single data entry and return issues if any."""
    issues = []
    
    try:
        # Check required keys
        required_keys = ['atomic_numbers', 'positions', 'energy']
        for key in required_keys:
            if key not in entry:
                issues.append(f"Missing key: {key}")
        
        if issues:
            return issues
            
        # Check atomic numbers
        atomic_numbers = entry['atomic_numbers']
        if not torch.is_tensor(atomic_numbers):
            atomic_numbers = torch.tensor(atomic_numbers)
        if atomic_numbers.dim() != 1 or len(atomic_numbers) == 0:
            issues.append(f"Invalid atomic_numbers shape: {atomic_numbers.shape}")
        if atomic_numbers.min() < 1 or atomic_numbers.max() > 118:
            issues.append(f"Invalid atomic numbers range: {atomic_numbers.min()}-{atomic_numbers.max()}")
            
        # Check positions
        positions = entry['positions']
        if not torch.is_tensor(positions):
            positions = torch.tensor(positions)
        if positions.dim() != 2 or positions.shape[1] != 3:
            issues.append(f"Invalid positions shape: {positions.shape}, expected [N, 3]")
        if positions.shape[0] != len(atomic_numbers):
            issues.append(f"Mismatch: {len(atomic_numbers)} atoms but {positions.shape[0]} positions")
        if torch.isnan(positions).any() or torch.isinf(positions).any():
            issues.append("NaN or Inf values in positions")
            
        # Check energy
        energy = entry['energy']
        if torch.is_tensor(energy):
            energy = energy.item()
        if not isinstance(energy, (int, float)) or torch.isnan(torch.tensor(energy)):
            issues.append(f"Invalid energy value: {energy}")
            
    except Exception as e:
        issues.append(f"Exception during validation: {str(e)}")
        
    return issues

def create_data_object(entry, idx):
    """Create a Data object with proper error handling."""
    try:
        # Convert to tensors with proper dtypes
        atomic_numbers = entry['atomic_numbers']
        if not torch.is_tensor(atomic_numbers):
            atomic_numbers = torch.tensor(atomic_numbers)
        atomic_numbers = atomic_numbers.long()
        
        positions = entry['positions']
        if not torch.is_tensor(positions):
            positions = torch.tensor(positions)
        positions = positions.to(DTYPE)
        
        energy = entry['energy']
        if torch.is_tensor(energy):
            energy = energy.item()
        
        # Create target tensor - shape [1] for single graph
        y = torch.tensor([energy], dtype=DTYPE)
        
        data = Data(
            z=atomic_numbers,
            pos=positions,
            y=y
        )
        
        return data, None
        
    except Exception as e:
        return None, f"Failed to create Data object for entry {idx}: {str(e)}"

def test_model_forward(model, sample_data):
    """Test if the model can perform a forward pass."""
    try:
        model.eval()
        print("model eval")
        with torch.no_grad():
            output = model.forward(sample_data.to(DEVICE))
        print(f"✓ Model forward pass successful. Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Model forward pass failed: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return False

# --- Data Loading and Pre-processing ---
print(f"Loading enriched dataset from {DATA_PATH}...")
if not os.path.exists(DATA_PATH):
    print(f"Error: Dataset not found at {DATA_PATH}. Please run 1_prepare_dataset.py and 2_enrich_dataset.py first.")
    sys.exit(1)

try:
    enriched_dataset_dicts = torch.load(DATA_PATH, map_location='cuda')
    print(f"✓ Successfully loaded {len(enriched_dataset_dicts)} entries")
except Exception as e:
    print(f"✗ Failed to load dataset: {str(e)}")
    sys.exit(1)

print("Validating and pre-processing dataset...")
dataset_list = []
validation_errors = []

for idx, entry in enumerate(enriched_dataset_dicts):
    # Validate entry
    issues = validate_data_entry(entry, idx)
    if issues:
        validation_errors.extend([f"Entry {idx}: {issue}" for issue in issues])
        continue
    
    # Create Data object
    data, error = create_data_object(entry, idx)
    if error:
        validation_errors.append(error)
        continue
        
    dataset_list.append(data)

if validation_errors:
    print(f"✗ Found {len(validation_errors)} validation errors:")
    for error in validation_errors[:10]:  # Show first 10 errors
        print(f"  - {error}")
    if len(validation_errors) > 10:
        print(f"  ... and {len(validation_errors) - 10} more errors")
    
    if len(dataset_list) == 0:
        print("No valid data entries found. Exiting.")
        sys.exit(1)
    else:
        print(f"Continuing with {len(dataset_list)} valid entries out of {len(enriched_dataset_dicts)}")

print(f"✓ Successfully processed {len(dataset_list)} graphs")

# Test a single data point
if dataset_list:
    sample_data = dataset_list[0]
    print(f"Sample data - Atoms: {len(sample_data.z)}, Positions shape: {sample_data.pos.shape}, Energy: {sample_data.y.item()}")

# Create DataLoader
try:
    loader = DataLoader(dataset_list, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✓ DataLoader created successfully")
except Exception as e:
    print(f"✗ Failed to create DataLoader: {str(e)}")
    sys.exit(1)
# --- Model, Optimizer, and Loss ---
print("Initializing model...")
try:
    # Initialize the model first (it will be float32 by default)
    model = model
    
    # THEN, convert all its parameters and buffers to your desired DTYPE
    model = model.to(dtype=DTYPE, device=DEVICE) 
    
    print(f"✓ Model loaded successfully on {DEVICE} with dtype={next(model.parameters()).dtype}")
    
    # ... rest of the code
    
except Exception as e:
    print(f"✗ Failed to initialize model: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

# Test model with sample data
if dataset_list:
    print("dataset:", dataset_list[0])
    if not test_model_forward(model, dataset_list[0]):
        print("Model forward pass failed. Please check your model implementation.")
        sys.exit(1)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# --- Training Loop ---
print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs...")
print("=" * 50)

try:
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, batch_data in enumerate(loader):
            try:
                # Move data to device
                batch_data = batch_data.to(DEVICE)
                
                # Debug: Print batch info for first few batches
                if epoch == 0 and batch_idx < 3:
                    print(f"  Batch {batch_idx}: {batch_data.num_graphs} graphs, "
                          f"target shape: {batch_data.y.shape}")
                
                optimizer.zero_grad()
                
                # Forward pass
                output = model(batch_data)
                
                # Debug: Check output shape
                if epoch == 0 and batch_idx < 3:
                    print(f"  Output shape: {output.shape}, Target shape: {batch_data.y.shape}")
                
                # Compute loss
                # Ensure shapes match for loss calculation
                if output.dim() == 2 and output.shape[1] == 1:
                    output = output.squeeze(1)  # [batch_size, 1] -> [batch_size]
                if batch_data.y.dim() == 2 and batch_data.y.shape[1] == 1:
                    target = batch_data.y.squeeze(1)  # [batch_size, 1] -> [batch_size]
                else:
                    target = batch_data.y
                
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # Progress indicator
                if batch_idx % max(1, len(loader) // 5) == 0:
                    print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.6f}")
                    
            except Exception as e:
                print(f"✗ Error in batch {batch_idx}: {str(e)}")
                traceback.print_exc()
                continue
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"  Average Loss: {avg_loss:.6f}")
            print("-" * 30)
        else:
            print("  No successful batches in this epoch!")
            
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
except Exception as e:
    print(f"✗ Training failed: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

print("Training complete!")

# --- Save the Trained Model ---
try:
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✓ Trained surrogate model saved to '{MODEL_SAVE_PATH}'")
except Exception as e:
    print(f"✗ Failed to save model: {str(e)}")