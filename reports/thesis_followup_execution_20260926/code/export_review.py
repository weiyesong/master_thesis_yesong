"""Export actual Claude public responses and tool evidence, never hidden reasoning."""
import json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from common import OUT,dump,sha

SESSION='5afb33d9-4dac-4ed1-8d17-2e1ec2a65b13'
TRANSCRIPT=Path('/home/yesong/.claude/projects/-workspace')/(SESSION+'.jsonl')


def main():
    rounds=[]
    for name in ['protocol','code','final']:
        raw=OUT/f'claude_{name}_raw.json';data=json.loads(raw.read_text())
        assert data['session_id']==SESSION and data['subtype']=='success' and not data['is_error']
        (OUT/f'claude_{name}_review.md').write_text(data['result'].strip()+'\n')
        rounds.append({'round':name,'request':f'claude_{name}_request.md','response_path':raw.name,'response_sha256':sha(raw),
                       **{key:data.get(key) for key in ['session_id','subtype','is_error','num_turns','duration_ms','duration_api_ms','total_cost_usd']}})
    evidence=[];tools=Counter();models=set();versions=set()
    for line in TRANSCRIPT.open():
        event=json.loads(line);content=event.get('message',{}).get('content',[])
        if event.get('version'):versions.add(event['version'])
        if event.get('message',{}).get('model'):models.add(event['message']['model'])
        if not isinstance(content,list):continue
        for block in content:
            if block.get('type') not in ['tool_use','tool_result']:continue
            evidence.append({'timestamp':event.get('timestamp'),'session_id':SESSION,'message_uuid':event.get('uuid'),'block':block})
            if block['type']=='tool_use':tools[block['name']]+=1
    with (OUT/'claude_evidence_trace.jsonl').open('w') as f:
        for item in evidence:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    calls={r['block']['id']:r for r in evidence if r['block']['type']=='tool_use'}
    index=[{'tool_use_id':key,'tool':r['block']['name'],'timestamp':r['timestamp'],
            'target':r['block']['input'].get('file_path',r['block']['input'].get('description',r['block']['input'].get('pattern','')))} for key,r in calls.items()]
    dump(OUT/'claude_tool_index.json',index)
    dump(OUT/'claude_session_metadata.json',{'provider':'actual Anthropic Claude Code CLI','session_id':SESSION,'cli_versions_observed':sorted(versions),'models_observed':sorted(models),
         'cli_path':'/home/yesong/.vscode-server/extensions/anthropic.claude-code-2.1.283-linux-x64/resources/native-binary/claude',
         'transcript_source':str(TRANSCRIPT),'transcript_sha256_at_export':sha(TRANSCRIPT),'exported_utc':datetime.now(timezone.utc).isoformat(),
         'public_blocks':len(evidence),'tool_calls':dict(tools),'rounds':rounds,'final_resume_request':'claude_final_resume_request.md',
         'interrupted_attempt_record':'claude_final_attempt1_status.json','final_exit':json.loads((OUT/'claude_final_exit.json').read_text()),
         'evidence_policy':'Only public tool_use/tool_result blocks plus public final responses are exported. No hidden reasoning blocks.',
         'review_scope':'independent selected original-array calculations, mathematical tests, control and claim review; not duplicate execution of all 100 objects'})
    print('Exported',len(evidence),'public tool blocks and three successful public reviews')


if __name__=='__main__':main()
