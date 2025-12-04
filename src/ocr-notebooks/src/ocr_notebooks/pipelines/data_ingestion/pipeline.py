"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  
from .nodes import download_files_from_hf, extract_samples

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func = download_files_from_hf,
            inputs = [
                "params:school_notebooks.repo_id",
                "params:school_notebooks.files",
                "params:school_notebooks.output_dir",                
            ],
            outputs = "raw_school_notebooks_paths",
            name="download_school_notebooks_node",
        ),
        
        Node(
            func=extract_samples,
            inputs=[
                "annotations_train_raw", 
                "params:data_preparation.images_base_dir",
                "params:data_preparation.n_pages",
                "params:data_preparation.model_dir",
                "params:data_preparation.cer_threshold",
                ],
            outputs="train_line_samples",
            name="coco_to_line_samples_train",
        ),
        Node(
            func=extract_samples,
            inputs=[
                "annotations_val_raw", 
                "params:data_preparation.images_base_dir",
                "params:data_preparation.n_pages",
                "params:data_preparation.model_dir",
                "params:data_preparation.cer_threshold",                
                ],
            outputs="val_line_samples",
            name="coco_to_line_samples_val",
        ),
        Node(
            func=extract_samples,
            inputs=[
                "annotations_test_raw", 
                "params:data_preparation.images_base_dir",
                "params:data_preparation.n_pages",
                "params:data_preparation.model_dir",
                "params:data_preparation.cer_threshold",                
                ],
            outputs="test_line_samples",
            name="coco_to_line_samples_test",
        ),
    ])
