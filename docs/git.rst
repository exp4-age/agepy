:orphan:

git tutorial
============

This is a short introduction to *git*, which is copied from the
`git tutorial`_ on the sharepoint.

Installation
------------

* **On Windows:**
  There are several options for git under Windows. 
  The Windows PowerShell already supports git, so no installation is really 
  needed. To test if git is available, open a PowerShell and type ``git``
  and press enter. If is not available, you can install it with winget::

    winget install --id Git.Git -e --source winget
  
  If you don't want to use the PowerShell, you can install the official
  standard toolbox (https://git-scm.com/), which gives you the "Git Bash"
  and "Git GUI". 
  
* **Linux** has git installed on almost all distributions. If it is not installed
  for some reason, take a look at https://git-scm.com/download/linux.

Execute the commands ::

    git config --global user.name "Your name"
    git config --global user.email "youremail@physik.uni-kassel.de"

with your name and E-Mail in order to setup your signature for the commits 
you will make.

Using git
---------

*Git* is a version control system, which means that it can keep track of the
changes you are making in a project folder (repository).

To setup a repository, create and go to a project folder with ::

    mkdir path/to/your/project
    cd path/to/your/project

and initialize it::

    git init

.. note::

    If you want to start using git, you probably already have files you want to
    add into a repository. However, a repository can not be initialized in a
    non-empty folder. To work around this, just create a new folder, initialize
    a repository and then copy the existing files/folders into this newly
    created repository.

With ::

    git add -A

you tell *git* to track all files present in the project folder.

.. note::

    You can track only specific files by dropping the ``-A`` and 
    providing the path to files or folders instead.

.. note::

    You can create a file called ``.gitignore`` inside the repository, where
    you can specify files and folders, which should not be tracked. This is
    useful if you have large data files or a bunch of unimportant files.

In order to save the state of the added files to the history of the
repository, create a commit with::

    git commit -m "Some descriptive message"

For more information on using *git* you can checkout the
`Learn Git Branching`_ interactive tutorial. This is a really nice way to
learn the basics (don't worry, you don't need to do all the chapters - the 
first few will already give you a good starting point).

Commonly used git commands
--------------------------

The following list of commands is not complete. It only provides a quick 
overview. For a full documentation see the `git docs`_ or use the 
command ``git help``.

* ::
    
    git init <repository name>

  Creates a new folder named <repository name> inside the current folder and 
  initializes a repository inside this folder.

* ::
    
    git add <filepattern>

  Adds some files into the local stage part of the repository. Patterns like \*
  or \*.py can be used here.

* ::
    
    git commit -m "<Message>"

  Commits the currently added files with a message into the local repository.

* ::
    
    git log

  Show the history of a repository.

* ::
    
    git status

  Show the current status of the repository. Will show files that have been 
  altered after the last commit.

* ::
    
    git diff

  Shows detailes about changes made to files.

* ::
    
    git push <remote> <branch>

  Uploads the given branch (usually master) of the repository to the given 
  remote (usually origin).

* ::
    
    git pull

  Downloads the current status of the repository from the configured remote.

* ::
    
    git clone <remote-address>

  Clones a remote repository into the current local folder.

* ::
    
    git checkout <branch/commit>

  Loads the status of the repository at a given commit or loads the given 
  branch.

* ::
    
    git remote <options>

  Configures or shows the remote repository for this local repository. Will be 
  configured automatically if the local repository was downloaded via git 
  clone.

Ressources
----------

* Sharepoint `git tutorial`_
* official `git docs`_
* Tutorial `Learn Git Branching`_
* Online repository storage `GitHub`_ / `GitLab`_

.. _git tutorial: https://sharepoint.uni-kassel.de/sites/fb10-exp4/wiki/AGE%20Wiki/git.aspx
.. _Learn Git Branching: https://learngitbranching.js.org/?locale=en_US
.. _git docs: https://git-scm.com/docs
.. _GitHub: https://github.com/
.. _GitLab: https://gitlab.uni-kassel.de/users/sign_in
