import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from retouch import Editor
import studio


class GenerationUITests(unittest.TestCase):
    def test_cli_and_browser_entry_points(self):
        with patch('retouch.start') as start:
            for args in ([],['--ui']):
                studio.main(args)
                start.assert_called_with(page='/studio')
            studio.main(['--retouch'])
            start.assert_called_with()
        with patch('builtins.input',return_value='0'):
            studio.main(['--menu'])

    def test_generation_defaults_options_and_repeat_submission(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            with Editor(('127.0.0.1',0),root) as server, patch('retouch.IMAGES',root), \
                    patch('retouch.settings',return_value={'output_dir':tmp}), patch('retouch.subprocess.Popen') as launch:
                data={'request':'a'*32,'prompt':'猫'}
                server.generate(data)
                args=launch.call_args.args[0]
                self.assertEqual(args[args.index('--max-rounds')+1],'1')
                self.assertEqual(args[args.index('--output-dir')+1],tmp)
                self.assertNotIn('--8k',args)
                self.assertNotIn('--candidates',args)
                server.jobs.clear()
                self.assertEqual(server.generate(data),data['request'])
                launch.assert_called_once()
                self.assertEqual(len(list((root/'new-jobs').iterdir())),1)
                data.update(request='b'*32,quality='8K',style='水彩',seed=str(2**63-1),rounds=3,aspect='16:9')
                server.generate(data)
                args=launch.call_args.args[0]
                for option in ('--8k','--style=水彩','16:9',str(2**63-1),'--prompt=猫'):
                    self.assertIn(option,args)
                for field,value in [('prompt','   '),('rounds',-1),('seed',str(2**63)),('aspect','bad')]:
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        server.generate({**data,field:value})
