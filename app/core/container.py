from app.dependencies.dependencies import get_embedder, get_vector_store
from app.pipeline.rag_pipeline import RAGPipeline
from app.pipeline.study_pipeline import StudyPipeline


class PipelineContainer:
    def __init__(self):
        embedder = get_embedder()
        vector_store = get_vector_store()

        self.rag_pipeline = RAGPipeline(vector_store, embedder)
        self.study_pipeline = StudyPipeline(self.rag_pipeline.retriever)


container = PipelineContainer()


def get_container():
    return container