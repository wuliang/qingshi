import time
start=time.time()

setup_args = {
    'name': 'qingshi',
    'version': '0.1',
    'url': 'http://qingshi.org',
    'description': 'A framework to study probabilistic distribution of Chinese nature language ',
    'long_description': 'Qingshi is framework to study probabilistic distribution of Chinese nature languag, in multiple levels, for all kinds of purposes, from information retrieval to grammar learning.',
    'author': 'Wu liang',
    'maintainer': 'Wu liang',
    'maintainer_email': 'www.wuliang.cn@gmail.com',
    'license': 'GNU Lesser General Public License (LGPL)',
    'packages': ['qingshi'],
#    'cmdclass': cmdclasses,
#    'data_files': data_files,
#    'scripts': scripts,
}

try:
    from setuptools import setup, Extension, Feature
except ImportError:
    from distutils.core import setup, Extension
else:
    setup_args['install_requires'] = ['redis', 'pyyaml' ]


extra = {}

try:
    from qingshi import scripts
except ImportError:
    print "EOOR"
    pass
else:
    extra['cmdclass'] = {
        'interact': scripts.InteractCommand,
        'feed': scripts.FeedCommand,
        'refine': scripts.RefineCommand,
        'reset': scripts.ResetCommand,
        'serve': scripts.ServeCommand,
        'dump': scripts.DumpCommand,
        'info': scripts.InfoCommand
    }
zargs = dict(setup_args.items() + extra.items())
setup(**zargs)

elapsed = time.time() - start
print 'elapsed seconds is: ', elapsed
