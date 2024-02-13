from setuptools import setup

setup(name = 'ptep_mbs',
      version = '0.1',
      description = '',
      url = '',
      author = 'Nicoletta Krachmalnicoff',
      author_email = 'nkrach@sissa.it',
      license = 'MIT',
      packages = ['ptep_mbs'],
      package_dir = {'ptep_mbs': 'ptep_mbs'},
      zip_safe = False,
      entry_points = {
        'console_scripts': [
            'litebird_mbs = ptep_mbs.pipeline:__main__'
        ]
      })
