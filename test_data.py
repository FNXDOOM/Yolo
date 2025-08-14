import os
import yaml

def test_dataset():
    print("Testing dataset configuration...")
    
    # Check data.yaml
    if os.path.exists("data.yaml"):
        print("✓ data.yaml exists")
        try:
            with open("data.yaml", 'r') as f:
                config = yaml.safe_load(f)
                print(f"✓ data.yaml loaded successfully")
                print(f"  Classes: {config.get('nc', 'Not found')}")
                print(f"  Names: {config.get('names', 'Not found')}")
                print(f"  Train path: {config.get('train', 'Not found')}")
                print(f"  Val path: {config.get('val', 'Not found')}")
        except Exception as e:
            print(f"✗ Error reading data.yaml: {e}")
    else:
        print("✗ data.yaml not found")
        return
    
    # Check if training directory exists and has images
    train_dir = "css-data/images/train"
    if os.path.exists(train_dir):
        images = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\n✓ Training directory found with {len(images)} images")
        
        # Check if labels directory exists
        labels_dir = "css-data/labels/train"
        if os.path.exists(labels_dir):
            labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            print(f"✓ Labels directory found with {len(labels)} label files")
            
            # Check first few label files
            print("\nChecking first 3 label files:")
            for i, label_file in enumerate(labels[:3]):
                label_path = os.path.join(labels_dir, label_file)
                try:
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                        if content:
                            lines = content.split('\n')
                            classes = [line.split()[0] for line in lines if line.strip()]
                            print(f"  {label_file}: classes {classes}")
                        else:
                            print(f"  {label_file}: empty")
                except Exception as e:
                    print(f"  {label_file}: error reading - {e}")
        else:
            print("✗ Labels directory not found")
    else:
        print("✗ Training directory not found")

if __name__ == "__main__":
    test_dataset()
