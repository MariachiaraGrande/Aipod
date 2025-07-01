This is a template of a package.

More info on packaging:
https://packaging.python.org/guides/distributing-packages-using-setuptools/

### Quick-start:

1. Rename the main folder <package_template> to the desired name
2. Modify variable MOD_NAME in the file <setup.py> with the desired package name
3. Modify MANIFEST.in where you see package_template with the name of the package
4. [optional] Update the version in <package_template/_version.py>

### Development:

- All functions/classes code goes <package_template>
- Scripts (executables) go in <scripts>

### Installation (from source)

For deployment:

```shell
foo@bar:package_template$ pip install . 
```

For development:

```shell
foo@bar:~/package_template $ pip install -e . 
```

### build

Source package:

```shell
foo@bar:~/package_template $ python setup.py sdist
```

Wheel package:

```shell
foo@bar:~/package_template $ pip install wheel
foo@bar:~/package_template $ python setup.py bdist_wheel
```