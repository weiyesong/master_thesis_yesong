import subprocess,json
from pathlib import Path
p=Path('/workspace/reports/thesis_followup_execution_20260926')
args=['/home/yesong/.vscode-server/extensions/anthropic.claude-code-2.1.283-linux-x64/resources/native-binary/claude','-p','--resume','5afb33d9-4dac-4ed1-8d17-2e1ec2a65b13','--permission-mode','dontAsk','--tools','Read,Glob,Grep,Bash','--allowedTools','Read,Glob,Grep,Bash','--output-format','json']
with (p/'claude_final_resume_request.md').open() as inp,(p/'claude_final_raw.json').open('w') as out,(p/'claude_final_stderr.log').open('w') as err:
    r=subprocess.run(args,stdin=inp,stdout=out,stderr=err,cwd='/workspace')
(p/'claude_final_exit.json').write_text(json.dumps({'returncode':r.returncode})+'\n')
