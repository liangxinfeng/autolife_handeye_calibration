from setuptools import setup
from glob import glob
name = 'autolife_handeye_calibration_306'
setup(name=name, version='0.2.0', packages=[name],
      data_files=[('share/ament_index/resource_index/packages', ['resource/'+name]),
                  ('share/'+name, ['package.xml', 'README_ZH.md', 'VERIFICATION_ZH.md', 'HD_AUTO_CALIBRATION_ZH.md']),
                  ('share/'+name+'/config', glob('config/*.yaml')),
                  ('share/'+name+'/launch', glob('launch/*.py'))],
      entry_points={'console_scripts': ['calibration_app = '+name+'.app:main']})
