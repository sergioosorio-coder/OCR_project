"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.1.1
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import load_trocr_model_and_processor, build_trocr_datasets, train_trocr_model, save_trocr_artifacts

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
            Node(
                func=load_trocr_model_and_processor,
                inputs="params:model_training.pretrained_model_name",
                outputs=dict(processor="trocr_processor", model="trocr_model"),
                name="load_trocr_model_and_processor",
            ),
            Node(
                func=build_trocr_datasets,
                inputs=[
                    "train_line_samples",
                    "val_line_samples",
                    "trocr_processor",
                    "params:model_training.max_target_length",
                ],
                outputs=dict(
                    train_dataset="trocr_train_dataset",
                    val_dataset="trocr_val_dataset",
                ),
                name="build_trocr_datasets",
            ),
            Node(
                func=train_trocr_model,
                inputs=[
                    "trocr_model",
                    "trocr_processor",
                    "trocr_train_dataset",
                    "trocr_val_dataset",
                    "params:model_training.training_args",
                ],
                outputs=dict(
                    trained_model="trocr_trained_model",
                    metrics="trocr_eval_metrics",
                ),
                name="train_trocr_model",
            ),
            Node(
                func=save_trocr_artifacts,
                inputs=[
                    "trocr_trained_model",
                    "trocr_processor",
                    "trocr_eval_metrics",
                    "params:model_training.output_dir",
                ],
                outputs="trocr_artifacts_info",
                name="save_trocr_artifacts",
            ),
    ])
