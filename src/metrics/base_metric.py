from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, pred_texts: list[str], target_texts: list[str]):
        """
        Defines metric calculation logic for a given target and predicted texts.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()
