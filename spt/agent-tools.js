// Page tool contracts. Never expose session credentials or embedded file data.
export function createAgentTools(context) {
  const attempts=new Map();
  const hash=async value=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value))),byte=>byte.toString(16).padStart(2,'0')).join('');
  const object={type:'object',properties:{},additionalProperties:false};
  const integer=(maximum)=>({type:'integer',minimum:0,...(maximum?{maximum}:{})});
  const ids={type:'array',items:{type:'string'},maxItems:20000};
  const fail=(code,message)=>{const error=new Error(message);error.code=code;throw error;};
  const bound=(value,fallback,max)=>{if(value===undefined)return fallback;if(!Number.isInteger(value)||value<0||value>max)fail('invalid_argument','Invalid page bounds.');return value;};
  const wrap=execute=>async input=>{try{return {ok:true,...await execute(input||{})};}catch(error){return {ok:false,error:{code:error.code||'operation_failed',message:error.message},workspace:context.workspace()};}};
  const tool=(name,description,inputSchema,execute,readOnly=false)=>({name,description,inputSchema,annotations:{readOnlyHint:readOnly,untrustedContentHint:true,consequentialHint:name==='submit_spt_question'},execute:wrap(execute)});
  function checkWorkspace(expected){if(expected!==context.workspace().workspaceId)fail('stale_workspace','The workspace changed. Read it again before acting.');}
  return [
    tool('read_spt_workspace','Read dataset counts, workspace ID, availability, projection readiness and selection. Does not send analysis or expose credentials.',object,()=>context.workspace(),true),
    tool('list_spt_tracks','List exact file-qualified IDs, raw/encoder/view availability and preview acquisition summaries (timing, units, optional z). Null means unknown; intervals are observed, not an inferred frame rate. Paginated; use these exact IDs for selection.',{...object,properties:{offset:integer(),limit:{type:'integer',minimum:1,maximum:200},query:{type:'string',maxLength:200}}},input=>{
      const offset=bound(input.offset,0,Number.MAX_SAFE_INTEGER),limit=bound(input.limit,100,200);
      if(!limit||input.query!==undefined&&(typeof input.query!=='string'||input.query.length>200))fail('invalid_argument','Use a text query and a limit from 1 to 200.');
      const query=(input.query||'').toLocaleLowerCase();const tracks=context.tracks().filter(t=>!query||t.id.toLocaleLowerCase().includes(query));
      return {workspaceId:context.workspace().workspaceId,total:tracks.length,offset,tracks:tracks.slice(offset,offset+limit),nextOffset:offset+limit<tracks.length?offset+limit:null};
    },true),
    tool('read_spt_analysis','Read one response, paginated answer/artifact metadata, and sync state/refresh timestamps. Check sync state before trusting cached status. No file payloads. Response indices are zero-based; omit to read latest.',{...object,properties:{responseIndex:integer(),answerOffset:integer(),answerLimit:{type:'integer',minimum:1,maximum:12000},artifactOffset:integer(),artifactLimit:{type:'integer',minimum:1,maximum:200}}},input=>{
      const analysis=context.analysis(),sync=context.workspace().sync||null,connectionWarning=context.workspace().connectionWarning||null,index=bound(input.responseIndex,analysis.turns.length-1,Number.MAX_SAFE_INTEGER);
      if(!analysis.turns.length)return {workspaceId:context.workspace().workspaceId,status:analysis.status,sync,connectionWarning,responseCount:0,response:null,artifacts:[]};
      if(index>=analysis.turns.length)fail('unknown_response','Response index is outside this analysis.');
      const offset=bound(input.answerOffset,0,Number.MAX_SAFE_INTEGER),limit=bound(input.answerLimit,6000,12000);if(!limit)fail('invalid_argument','Answer limit must be positive.');
      const artifactOffset=bound(input.artifactOffset,0,Number.MAX_SAFE_INTEGER),artifactLimit=bound(input.artifactLimit,100,200);if(!artifactLimit)fail('invalid_argument','Artifact limit must be positive.');
      const turn=analysis.turns[index],answer=turn.answer||'';
      return {workspaceId:context.workspace().workspaceId,status:analysis.status,sync,connectionWarning,responseCount:analysis.turns.length,response:{index,status:turn.status,question:turn.question,selection:turn.selection||null,answer:answer.slice(offset,offset+limit),answerOffset:offset,answerLength:answer.length,nextAnswerOffset:offset+limit<answer.length?offset+limit:null},artifactCount:analysis.artifacts.length,artifactOffset,artifacts:analysis.artifacts.slice(artifactOffset,artifactOffset+artifactLimit),nextArtifactOffset:artifactOffset+artifactLimit<analysis.artifacts.length?artifactOffset+artifactLimit:null};
    },true),
    tool('stage_spt_sample','Stage demo dataset and question without sending analysis. Returns summary; preserves an existing draft.',object,async()=>{await context.stageSample();return context.workspace();}),
    tool('set_spt_projection','Show raw trajectories, PCA, UMAP or t-SNE. Returns actual resulting projection/readiness, including fallback or cancellation.',{...object,properties:{method:{type:'string',enum:['raw','pca','umap','tsne']}},required:['method']},async input=>({result:await context.project(input.method),workspace:context.workspace()})),
    tool('select_spt_tracks','Select exact file-qualified IDs. Unknown or out-of-view tracks are rejected. Empty array clears selection.',{...object,properties:{ids},required:['ids']},async input=>{await context.select(input.ids);return context.workspace();}),
    tool('submit_spt_question','Send a requested question for the current dataset/selection; may start cloud computation. First read workspace. Supply its workspaceId and selectedTrackIds. Use a unique requestId, reused only for retry of identical arguments within this tab; receipt metadata survives reload where storage is available. Conflicting human drafts are preserved.',{...object,properties:{question:{type:'string',minLength:1,maxLength:40000},expectedWorkspaceId:{type:'string'},expectedSelectedTrackIds:ids,requestId:{type:'string',minLength:1,maxLength:128}},required:['question','expectedWorkspaceId','expectedSelectedTrackIds','requestId']},async input=>{
      if(typeof input.question!=='string'||!input.question.trim()||input.question.length>40000||typeof input.requestId!=='string'||!input.requestId.trim()||input.requestId.length>128||!Array.isArray(input.expectedSelectedTrackIds)||input.expectedSelectedTrackIds.length>20000||input.expectedSelectedTrackIds.some(id=>typeof id!=='string'))fail('invalid_argument','Provide a question, request ID and exact selected-track IDs.');
      const [key,signature]=await Promise.all([hash(input.requestId),hash(JSON.stringify([input.expectedWorkspaceId,input.question,[...new Set(input.expectedSelectedTrackIds)].sort()]))]);
      const prior=attempts.get(key);
      if(prior){if(prior.signature!==signature)fail('request_id_conflict','This request ID was already used for different arguments.');if(prior.promise)return {...await prior.promise,workspace:context.workspace(),reused:true};}
      const receipt=context.receipts?.get(key);
      if(receipt){if(receipt.signature!==signature)fail('request_id_conflict','This request ID was already used for different arguments.');if(receipt.submission)return {submission:receipt.submission,workspace:context.workspace(),reused:true,retryProtection:'tab-session'};}
      checkWorkspace(input.expectedWorkspaceId);
      const state=context.workspace(),wanted=[...new Set(input.expectedSelectedTrackIds)].sort(),actual=[...state.selectedTrackIds].sort();
      if(JSON.stringify(wanted)!==JSON.stringify(actual))fail('selection_changed','The selection changed. Read the workspace before sending.');
      if(state.sending||state.loading)fail('workspace_busy','Wait for the current load or send to finish.');
      const draft=context.draft();if(draft.trim()&&draft!==input.question)fail('draft_conflict','A different human draft is present. It was preserved.');
      // Bound retained successful request IDs; never evict an in-flight request.
      if(!attempts.has(key)&&attempts.size>=128)fail('request_limit','This page session has reached 128 tool submissions. Reload to start a new tool session.');
      context.receipts?.save(key,signature,null);
      const record={signature,promise:null};attempts.set(key,record);
      record.promise=Promise.resolve().then(()=>context.submit(input.question,{requestId:'agent-'+key})).then(result=>{const persisted=context.receipts?.save(key,signature,result)||false;return {submission:result,workspace:context.workspace(),reused:false,retryProtection:persisted?'tab-session':'page-session'};}).catch(error=>{record.promise=null;throw error;});
      return record.promise;
    })
  ];
}
