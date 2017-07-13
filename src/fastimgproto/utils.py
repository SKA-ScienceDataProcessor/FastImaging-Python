

def reset_progress_bar(progress_bar, total, description="Doing something"):
    """
    Short subroutine to reset a progress bar.

    Currently setup for tqdm progress bars, but we could imagine this
    function evolving for a different libary or even handling progress bars
    polymorphically.

    Mainly it just tidies 3 irrelevant lines into one clear function call,
    which is less distracting from key logic.

    Currently
    Args:
        progress_bar (tqdm.tqdm): Progress display class
        total (int): total number of iterations being stepped through
        description (str):  Description to display

    Returns:

    """
    progress_bar.n = 0
    progress_bar.total=total
    progress_bar.desc=description
    return
