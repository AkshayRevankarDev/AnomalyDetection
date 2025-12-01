from src.engine.classifier_trainer import ClassifierTrainer

if __name__ == "__main__":
    # The downloaded dataset structure is likely data/raw/brain_mri_4class/Training and Testing
    # Let's verify the path first. The unzip command extracted to data/raw/brain_mri_4class
    
    data_path = "data/raw/brain_mri_4class"
    
    trainer = ClassifierTrainer(data_path=data_path, epochs=10)
    trainer.train()
