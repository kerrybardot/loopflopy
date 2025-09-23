import sys
import os


class Project:
    """
    A project management class for organizing LoopFlopy modeling workflows.
    
    This class provides a structured approach to managing groundwater modeling
    projects by organizing file paths, executable locations, and automatically
    creating necessary directory structures for model outputs and results.
    
    Parameters
    ----------
    name : str
        Project name identifier. Used for naming model files and outputs.
    workspace : str
        Relative (or absolute) path to the main working directory where model files will be stored.
        This is typically where MODFLOW input files and temporary files are written.
    results : str
        Relative (or absolute) path to the directory for storing model results and output files.
        This includes MODFLOW output files, heads, budgets, and post-processing results.
    figures : str
        Relative (or absolute) path to the directory for storing generated plots and figures.
        All visualization outputs from plotting methods will be saved here.
    triexe : str
        Relative (or absolute) path to the Triangle executable for mesh generation.
        Required for creating unstructured triangular and Voronoi meshes.
    mfexe : str
        Relative (or absolute) path to the MODFLOW 6 executable (mf6.exe).
        Required for running groundwater flow simulations.
    
    Attributes
    ----------
    name : str
        Project name identifier.
    workspace : str
        Working directory path.
    results : str
        Results directory path.
    figures : str
        Figures directory path.
    triexe : str
        Triangle executable path.
    mfexe : str
        MODFLOW 6 executable path.
    
    Notes
    -----
    The class automatically creates the workspace, results, and figures directories
    if they don't already exist. This ensures that the project structure is ready
    for model development and output generation.
    
    Directory Structure:
    - workspace/: Model input files, temporary files, mesh generation files
    - results/: MODFLOW output files, heads, budgets, flow results
    - figures/: PNG files from plotting and visualization methods
    
    Examples
    --------
    >>> # Create a new project with standard directory structure
    >>> project = Project(
    ...     name='test_model',
    ...     workspace='../modelfiles',
    ...     results='../results', 
    ...     figures='../figures',
    ...     triexe='../exe/triangle.exe',
    ...     mfexe='../exe/mf6.exe'
    ... )
    
    >>> # Project automatically creates directories if they don't exist
    >>> print(f"Working in: {project.workspace}")
    >>> print(f"Results saved to: {project.results}")
    
    See Also
    --------
    Mesh : For creating computational meshes using triexe
    Flowmodel : For running MODFLOW simulations using mfexe
    """
    def __init__(self, name, workspace, results, figures, triexe, mfexe):
        """
        Initialize a new Project with specified paths and executables.
        
        Parameters
        ----------
        name : str
            Project name identifier.
        workspace : str
            Path to working directory for model files.
        results : str
            Path to results directory for outputs.
        figures : str
            Path to figures directory for plots.
        triexe : str
            Path to Triangle executable.
        mfexe : str
            Path to MODFLOW 6 executable.
        
        Notes
        -----
        Automatically creates workspace, results, and figures directories
        if they don't exist using os.makedirs() with exist_ok=True.
        """

        self.name = name
        self.workspace = workspace
        self.results = results
        self.figures = figures
        self.triexe = triexe
        self.mfexe = mfexe
        
        if not os.path.isdir(self.workspace): os.makedirs(self.workspace, exist_ok=True)
        if not os.path.isdir(self.results):   os.makedirs(self.results, exist_ok=True)
        if not os.path.isdir(self.figures):   os.makedirs(self.figures, exist_ok=True)

