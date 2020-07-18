import os
import shelve
import click


def create_log(f_name, save_dir=None):
    if save_dir is None:
        save_dir = os.path.dirname(f_name)

    cmd = f"cp {f_name} {save_dir}/.run.{os.path.basename(f_name)}"
    print(cmd)
    os.system(cmd)
    return


def save_parameters(f_save, hidden=True):#, to_save=True):
    """ Save paraemters to a pickled file"""
    print(f_save)
    f_save = str(f_save)+ ".parameters"
    print('new f_save', f_save)
    params = shelve.open(f_save, 'n')  # 'n' for new
    for key in dir():
        print('key', key)
        try:
            params[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    params.close()
    print('here')
    return


def visualize_parameters(dirs=(), param_files=(), save_dir=None):
    return


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("f_save", type=click.Path())
@click.option("--save_dir", default="")
def main(f_save, save_dir):
    if save_dir == "":
        save_dir = None
    #create_log(f_save, save_dir=save_dir)
    save_parameters(f_save, hidden=True)
    return


if __name__ == "__main__":
    main()

