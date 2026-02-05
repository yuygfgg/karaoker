from __future__ import annotations

from abc import ABC, abstractmethod

from karaoker.pipeline_types import AudioAssets, PipelineConfig, PipelinePaths, TranscriptResult


class TranscriptProvider(ABC):
    @abstractmethod
    def transcribe(
        self, *, audio: AudioAssets, paths: PipelinePaths, config: PipelineConfig
    ) -> TranscriptResult:
        raise NotImplementedError
