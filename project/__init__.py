import argparse
import mamba_ssm
import subprocess
from bonito.cli import basecaller, download, duplex, evaluate, export, train, view

'''
# These lines came from an example provided by ChatGPT, their necessity will be determined later.
def install_package(package_name):
    subprocess.run(['mamba', 'install', package_name], check=True)

install_package()
'''
modules= [
    'basecall', 'download', ''' 'duplex', ''' 'evaluate', 'export', 'train', 'view',
    ]

'''
Explanation:
basecall- TBD
download-TBD
duplex-TBD, I'm not even sure we need it
evaluate- TBD
export- TBD
train- training the basecaller
view- TBD

'''

__bonito__version__= '0.8.1'
__Tama__version__='0.1.0'

#Until our basecaller will have a constant name, Tama is the temporary name.

def main():
    
    parser = argparse.ArgumentParser('versions', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-v', '--version', action='version',
        version='%(prog)s Bonito version: {} | Tama version: {}'.format(__bonito__version__, __Tama__version__)
    )

    subparsers = parser.add_subparsers(
        title='subcommands', description='valid commands',
        help='additional help', dest='command'
    )
    subparsers.required = True

    for module in modules:
        mod = globals()[module]
        p = subparsers.add_parser(module, parents=[mod.argparser()])
        p.set_defaults(func=mod.main)

    args = parser.parse_args()
    args.func(args)

