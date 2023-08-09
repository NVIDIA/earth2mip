from typing import Protocol, List, Iterator, Tuple, Any, Optional
import datetime
import torch
from earth2mip.schema import Grid


ChannelNameT = str


class TimeLoop(Protocol):
    """Abstract protocol that a custom time loop must follow

    This is a callable which yields time and output information. Some attributes
    are required to define the input and output data required.

    The expectation is that this class and the data passed to it are on the same
    device. While torch modules can be moved between devices easily, this is not
    true for all frameworks.
    """

    in_channel_names: List[ChannelNameT]
    out_channel_names: List[ChannelNameT]
    grid: Grid
    n_history_levels: int = 1
    history_time_step: datetime.timedelta = datetime.timedelta(hours=0)
    time_step: datetime.timedelta
    device: torch.device

    def __call__(
        self, time: datetime.datetime, x: torch.Tensor, restart: Optional[Any] = None
    ) -> Iterator[Tuple[datetime.datetime, torch.Tensor, Any]]:
        """
        Args:
            x: an initial condition. has shape (B, n_history_levels,
                len(in_channel_names), Y, X).  (Y, X) should be consistent with
                ``grid``.
            time: the datetime to start with
            restart: if provided this restart information (typically some torch
                Tensor) can be used to restart the time loop

        Yields:
            (time, output, restart) tuples. ``output`` is a tensor with
                shape (B, len(out_channel_names), Y, X) which will be used for
                diagnostics. Restart data should encode the state of the time
                loop.
        """
        pass
