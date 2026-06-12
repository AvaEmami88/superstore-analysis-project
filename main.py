from report_pipeline import ReportPipeline


pipeline = ReportPipeline("Superstore.csv")
results = pipeline.run_all()
pipeline.save_results()
pipeline.generate_csv()





