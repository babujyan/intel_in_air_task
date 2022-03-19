import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='intel_in_air_task',
    version='0.1',
    author='babujyan',
    author_email='babujyan.hrach@hotmail.com',
    description='Task 2 - Field State Classification:',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/babujyan/intel_in_air_task',
    project_urls={
        "Bug Tracker": "https://github.com/babujyan/intel_in_air_task/issues"
        },
    packages=['intel_in_air_task']
    )
