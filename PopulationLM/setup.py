from setuptools import find_packages, setup
setup(name="foo",
      version="0.1",
      description="A foo utility",
      author="Ewen Cheslack-Postava",
      author_email='me@ewencp.org',
      platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
      license="BSD",
      url="http://github.com/ewencp/foo",
      packages=find_packages(),
      )
