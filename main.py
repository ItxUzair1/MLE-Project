from src.pipeline.training_pipeline import TrainingPipeline


def main():
    print("Executing Telco Churn Training Pipeline...")
    pipeline = TrainingPipeline()
    results = pipeline.run_pipeline()
    print("Pipeline Execution Completed Successfully!")
    print(f"Results Summary: {results}")


if __name__ == "__main__":
    main()
