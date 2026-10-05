import {SPT_API_ORIGIN} from "./deployment-config.js?v=c34f6f9e1a47544b";
import {createCloudApi,cloudPollDelay} from "./cloud-client.js?v=c34f6f9e1a47544b";
import {renderAnswer,resolveArtifactLink} from './answer-renderer.js?v=c34f6f9e1a47544b';
import {ProjectorClient, readEmbeddings, questionForDisplay, pickTrajectory, selectSavedJob} from './projector-client.js?v=c34f6f9e1a47544b';
import {createColoring} from './coloring.js?v=c34f6f9e1a47544b';
const MAX_UPLOAD_BYTES=128*1024*1024;
const $ = id => document.getElementById(id);
const main = document.querySelector('main');
let files = [], active = null, latest = null, points = [], selected = new Set(), polygon = [], drawing = false;
let artifactUrls = [], pendingFiles=[], started=false, autoViewVersion=null, selectedCardId=null;
let embedding = null, embeddingId = null, projectionWorker = null, projectedEmbedding = null;
let embeddingArtifactsVersion = null;
let finishProjection = null;
let legacyApi=null, legacyVersion=null, legacyMode=false, projectionGeneration=0;
let pendingFollowup = null;
let pointColoring=null,colorVersion='null';
let pollTimer, sending = false, lastSignature = '', creationKey = crypto.randomUUID() + crypto.randomUUID();
let sampleFile=null, sampleContext='';
let connectionError=null;
let projectorClientId;
try { projectorClientId=sessionStorage.getItem('spt.projector.client'); } catch {}
if(!/^[0-9a-f]{32}$/.test(projectorClientId||''))projectorClientId=crypto.randomUUID().replaceAll('-','');
try { sessionStorage.setItem('spt.projector.client',projectorClientId); } catch {}
const projectorBridge=new ProjectorClient({api,context:()=>active,state:projectorState,project:setProjection,select:selectTracks,color:setColoring,
  clientId:projectorClientId,storage:{getItem:key=>sessionStorage.getItem(key),setItem:(key,value)=>sessionStorage.setItem(key,value)}});
function savedJobs() { try { const value = JSON.parse(localStorage.getItem('spt.jobs') || '[]'); return Array.isArray(value) ? value.filter(j => typeof j.id === 'string' && typeof j.token === 'string' && (!j.expiresAt || j.expiresAt*1000>Date.now())) : []; } catch { return []; } }
const saved = savedJobs();
const fragment = new URLSearchParams(location.hash.slice(1));
if (fragment.has('job') && fragment.has('key')) {
  active = {id: fragment.get('job'), token: fragment.get('key')};
  remember(active);
  history.replaceState(null, '', './#job=' + active.id);
} else {
  active = selectSavedJob(saved,fragment);
}

function remember(job) {
  const all = savedJobs().filter(j => j.id !== job.id);
  all.push({...job,title:latest?.turns?.[0]?.question?.slice(0,70)||job.title||'New conversation'});
  localStorage.setItem('spt.jobs', JSON.stringify(all));
  renderHistory();
}
function renderHistory() {
  const options = [Object.assign(document.createElement('option'), {value:'', textContent:'Choose an analysis'})];
  for (const job of savedJobs().reverse()) options.push(Object.assign(document.createElement('option'), {value:job.id, textContent:job.id.slice(0,10)}));
  $('history').replaceChildren(...options); $('history').value = active?.id || '';
  $('conversation-list').replaceChildren(...savedJobs().reverse().map(job=>{const b=document.createElement('button');b.textContent=job.title||'Analysis '+job.id.slice(0,10);b.className=active?.id===job.id?'current':'';b.onclick=()=>{$('history').value=job.id;$('history').onchange();sidebar(false);};return b;}));
}
$('history').onchange = async () => { const job=savedJobs().find(j=>j.id===$('history').value); if(!job)return; clearTimeout(pollTimer); active=job;error(null);latest=null;lastSignature='';pendingFiles=[];started=false;resetEmbedding();setFiles([]);showJourney();history.replaceState(null,'','./#job='+job.id); await refresh(); };
function sidebar(open){$('session-sidebar').hidden=!open;$('toggle-sidebar').setAttribute('aria-expanded',String(open));}
$('toggle-sidebar').onclick=()=>sidebar($('session-sidebar').hidden);$('close-sidebar').onclick=()=>sidebar(false);
function showJourney(){
  const talking=!!active||started;main.classList.toggle('onboarding',!talking);$('welcome').hidden=talking;
  const destination=talking?$('data-panel'):$('welcome-upload');
  if($('upload-box').parentElement!==destination)destination.append($('upload-box'));
  if(!talking){panels(true,true);$('send-hint').textContent='Include position units and time between frames, or try demo data.';}
}
const cloudApi=createCloudApi({origin:SPT_API_ORIGIN,encode,onUploadProgress:(sent,total)=>{$('status').textContent='Uploading '+Math.round(sent/Math.max(1,total)*100)+'%…';},onExpiry:id=>{
  const all=savedJobs().filter(j=>j.id!==id);localStorage.setItem('spt.jobs',JSON.stringify(all));renderHistory();
}});
async function api(path,data,auth=active){
  const result=await cloudApi(path,data,auth);
  if(result.id===active?.id&&result.expiresAt){active.expiresAt=result.expiresAt;remember(active);}
  return result;
}

function error(err) { $('error').textContent = err?.message || '';connectionError=null; }
function projectorState() {
  const state=legacyMode&&legacyApi?legacyApi.state():{projection:$('projection').value,projectionReady:projectionWorker===null,
    availableProjections:[...$('projection').options].filter(o=>!o.disabled).map(o=>o.value),
    selectedTrackIds:[...selected],selectedCount:selected.size,pointCount:(projectedEmbedding||points).length};
  return {...state,embeddingPublications:embedding?.publications||[],coloring:pointColoring?{label:pointColoring.spec.label,kind:pointColoring.spec.kind,...(pointColoring.spec.palette?{palette:pointColoring.spec.palette}:{}),mappedCount:pointColoring.mappedCount}:null};
}
function setColoring(spec){
  pointColoring=createColoring(spec,[...new Set([...points.map(p=>p.key),...(embedding?.points||[]).map(p=>p.id)])]);
  legacyApi?.color(pointColoring?[...pointColoring.colors]:null);draw();
  const legend=$('color-legend');legend.replaceChildren();legend.hidden=!pointColoring;
  if(!pointColoring)return;
  const title=document.createElement('strong');title.textContent=spec.label+(spec.units?' ('+spec.units+')':'');legend.append(title);
  const items=document.createElement('div');items.className='color-entries';
  for(const entry of pointColoring.entries){const item=document.createElement('span'),swatch=document.createElement('i');swatch.style.background=entry.color;item.append(swatch,document.createTextNode(entry.label));items.append(item);}
  if(spec.kind==='numeric'){const ramp=document.createElement('div');ramp.className='color-ramp';ramp.style.background='linear-gradient(90deg,'+pointColoring.ramp.join(',')+')';ramp.title=pointColoring.spec.palette+' · higher values are lighter';legend.append(ramp);}
  legend.append(items);const note=document.createElement('small');note.textContent=pointColoring.mappedCount+' trajectories mapped'+(pointColoring.missingCount?' · '+pointColoring.missingCount+' without values (gray)':'');legend.append(note);
}
function selectTracks(ids) {
  const knownIds=new Set([...points.map(t=>t.key),...(embedding?.points||[]).map(p=>p.id)]);
  if(!Array.isArray(ids)||ids.length>20000||ids.some(id=>typeof id!=='string'||!knownIds.has(id))) {
    const failure=new Error('Unknown track ID');failure.code='unknown_track';throw failure;
  }
  if(legacyMode&&legacyApi){
    try{return legacyApi.select(ids);}catch{const failure=new Error('Unknown or filtered track ID');failure.code='unknown_track';throw failure;}
  }
  selected=new Set(ids);draw();showTrajectory(ids);return {selectedTrackIds:[...selected]};
}
function decode(data) { return new TextDecoder().decode(Uint8Array.from(atob(data), c => c.charCodeAt(0))); }
function encode(bytes) {
  let text = '';
  for (let i = 0; i < bytes.length; i += 8192) text += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(text);
}
function setFiles(value) {
  files = value;
  $('dataset-count').textContent=files.length?`${files.length} file${files.length===1?'':'s'}`:'Add trajectories';
  $('data-panel').open=!files.length;
  renderFileControls();
  preview();showJourney();
}
function renderFileControls() {
  const committed=!!(active&&latest?.files?.length);
  $('files').disabled=$('sample').disabled=sending||committed;
  $('dataset-lock').hidden=!committed;
  const storedDemo=latest?.turns?.find(t=>t.question.includes('\n\nDemo dataset context:\n'))?.question.split('\n\nDemo dataset context:\n')[1];
  $('demo-info').hidden=!(sampleFile&&files.includes(sampleFile))&&!storedDemo;
  if(storedDemo)$('demo-context').replaceChildren(renderAnswer(storedDemo));
  $('file-list').replaceChildren(...files.map(file => {
    const chip = document.createElement('span'); chip.className = 'file-chip';chip.append(document.createTextNode(file.name));
    if(!sending&&(!active||!latest?.files?.length)){
      const remove=document.createElement('button');remove.type='button';remove.textContent='×';remove.setAttribute('aria-label','Remove '+file.name);
      remove.onclick=()=>{if(sending||active&&latest?.files?.length)return;const remaining=files.filter(f=>f!==file);if(active)pendingFiles=remaining;setFiles(remaining);error(null);};chip.append(remove);
    }
    return chip;
  }));
}
function resetEmbedding() {
  pointColoring=null;colorVersion='null';$('color-legend').hidden=true;
  autoViewVersion=null;selectedCardId=null;$('trajectory-card').hidden=true;
  projectionGeneration++;legacyApi?.cancel();legacyApi=null;legacyVersion=null;legacyMode=false;
  $('legacy-projector').removeAttribute('src');$('legacy-projector').hidden=true;$('plot').hidden=false;
  $('legacy-projector').parentElement.classList.remove('legacy');
  cancelProjection(); embedding=null; embeddingId=null; projectedEmbedding=null;
  embeddingArtifactsVersion=null;
  $('projection').value='raw';
  for (const option of $('projection').options) option.disabled=option.value!=='raw';
  $('embedding-info').textContent='Encoder projections appear when available.';
}
function cancelProjection() {
  projectionWorker?.terminate(); projectionWorker=null;
  finishProjection?.({cancelled:true}); finishProjection=null;
}
async function setProjection(method) {
  if(!['raw','pca','umap','tsne'].includes(method))throw new Error('Unknown projection.');
  updatePlotHelp(method);
  const request=++projectionGeneration;
  legacyApi?.cancel();
  if(method==='raw'){
    legacyMode=false;$('legacy-projector').hidden=true;$('plot').hidden=false;
    $('legacy-projector').parentElement.classList.remove('legacy');
    return setCompactProjection(method);
  }
  if(!embedding)throw new Error('Encoder vectors are not available yet.');
  cancelProjection();
  const frame=$('legacy-projector');frame.hidden=false;$('plot').hidden=true;
  frame.parentElement.classList.add('legacy');$('plot-empty').hidden=true;
  $('plot-meta').textContent='Loading the SPT projector…';
  try{
    if(!frame.getAttribute('src'))frame.src='./legacy-projector.html?v=c34f6f9e1a47544b';
    const deadline=Date.now()+25000;
    while(!frame.contentWindow?.sptLegacy&&Date.now()<deadline){
      if(request!==projectionGeneration)return {cancelled:true};
      await new Promise(resolve=>setTimeout(resolve,50));
    }
    if(!frame.contentWindow?.sptLegacy)throw new Error('SPT projector unavailable.');
    legacyApi=frame.contentWindow.sptLegacy;
    await legacyApi.ready;
    if(request!==projectionGeneration)return {cancelled:true};
    legacyMode=true;
    legacyApi.onChange=snapshot=>{
      if(!legacyMode)return;
      selected=new Set(snapshot.selectedTrackIds);
      $('projection').value=snapshot.projection;
      updatePlotHelp(snapshot.projection);
      $('selection').textContent=selected.size?selected.size+' selected':'All trajectories';
      $('plot-meta').textContent=snapshot.pointCount+' trajectories';
      showTrajectory(snapshot.selectedTrackIds);
    };
    if(legacyVersion!==embeddingId){
      const version=embeddingId, usedApi=legacyApi;
      const rawById=new Map(points.map(p=>[p.key,p]));
      const payload={points:embedding.points.map(p=>({...p,xPosition:rawById.get(p.id)?.path.map(v=>v[0])||[],yPosition:rawById.get(p.id)?.path.map(v=>v[1])||[]}))};
      await usedApi.load(payload);
      if(request!==projectionGeneration||version!==embeddingId)return {cancelled:true};
      legacyVersion=version;
    }
    if(request!==projectionGeneration)return {cancelled:true};
    const result=await legacyApi.project(method);
    if(request!==projectionGeneration||result.cancelled)return {cancelled:true};
    legacyApi.color(pointColoring?[...pointColoring.colors]:null);
    $('projection').value=method;updatePlotHelp(method);$('plot-title').textContent='SPT projector';$('projection-settings').hidden=false;
    return result;
  }catch(failure){
    if(request!==projectionGeneration)return {cancelled:true};
    legacyMode=false;legacyApi?.cancel();frame.hidden=true;$('plot').hidden=false;frame.parentElement.classList.remove('legacy');
    $('embedding-info').textContent='Original projector unavailable; showing the compact projection.';
    return setCompactProjection(method);
  }
}
async function setCompactProjection(method) {
  if (!['raw','pca','umap','tsne'].includes(method)) throw new Error('Unknown projection.');
  if (method!=='raw'&&!embedding) throw new Error('Encoder vectors are not available yet.');
  cancelProjection(); projectedEmbedding=null;
  $('projection').value=method;
  if(method==='raw') { $('plot-title').textContent='Trajectory preview'; preview(true); return {projection:'raw'}; }
  $('plot-title').textContent=method==='tsne'?'t-SNE':method.toUpperCase();
  $('plot-meta').textContent='Computing in your browser…';
  const worker=new Worker(new URL('./projection-worker.js?v=c34f6f9e1a47544b', import.meta.url),{type:'module'}); projectionWorker=worker;
  return new Promise((resolve,reject)=>{
    finishProjection=resolve;
    worker.onmessage=event=>{
      if(projectionWorker!==worker)return;
      const msg=event.data;
      if(msg.kind==='error'){worker.terminate();projectionWorker=null;finishProjection=null;error(new Error(msg.message));reject(new Error(msg.message));return;}
      if(msg.points){projectedEmbedding=msg.points;draw();}
      if(msg.kind==='progress')$('plot-meta').textContent='Computing · iteration '+msg.iteration;
      if(msg.kind==='complete'){
        $('plot-meta').textContent=msg.points.length+' trajectories · encoder vectors';
        $('plot-empty').hidden=true;worker.terminate();projectionWorker=null;finishProjection=null;
        resolve({projection:method,points:msg.points.length});
      }
    };
    worker.onerror=event=>{if(projectionWorker!==worker)return;worker.terminate();projectionWorker=null;finishProjection=null;reject(new Error(event.message||'Projection worker failed.'));};
    worker.postMessage({payload:embedding,method});
  });
}
$('projection').onchange=()=>setProjection($('projection').value).catch(error);
$('files').addEventListener('change', async event => {
  try {

    const chosen = [...event.target.files];event.target.value='';
    if(!chosen.length)return;
    if(sending||active&&latest?.files?.length)throw new Error('Start a new analysis for a different dataset.');
    const names=new Set(files.map(f=>f.name.toLowerCase()));
    for(const f of chosen){if(names.has(f.name.toLowerCase()))throw new Error('A file named '+f.name+' is already selected. Remove it first to replace it.');names.add(f.name.toLowerCase());}
    const selectedBytes=files.reduce((n,f)=>n+(f.bytes??Math.floor(f.data.length*3/4)-(f.data.endsWith('==')?2:f.data.endsWith('=')?1:0)),0);
    if(files.length+chosen.length>12||selectedBytes+chosen.reduce((n,f)=>n+f.size,0)>MAX_UPLOAD_BYTES)throw new Error('Choose up to 12 files, totaling at most '+MAX_UPLOAD_BYTES/1024/1024+' MiB.');
    const added=[];
    for(const f of chosen){const bytes=new Uint8Array(await f.arrayBuffer());const text=new TextDecoder('utf-8',{fatal:true}).decode(bytes);if(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/.test(text))throw new Error('Choose UTF-8 text files.');added.push({name:f.name,bytes:f.size,data:encode(bytes)});}
    const combined=[...files,...added];if(active)pendingFiles=combined;setFiles(combined);
    error(null);
  } catch (err) { error(err); }
});
async function loadSample(stageQuestion=true) {
  if(sending&&stageQuestion)throw new Error('Wait for your request to finish sending.');
  if (files.length) throw new Error('Start a new analysis before replacing your dataset.');
  const questionBefore=$('question').value,jobBefore=active?.id;
  const sample = await api('/api/sample', undefined, null);
  if(active?.id!==jobBefore||files.length)throw new Error('The dataset changed while the demo was loading. Try again.');
  sampleFile=sample.files[0];sampleContext=sample.question.split('## The question')[0];
  $('demo-context').replaceChildren(renderAnswer(sampleContext));
  if(active)pendingFiles=sample.files;setFiles(sample.files);
  if(stageQuestion&&!questionBefore.trim()&&!$('question').value.trim())$('question').value='Use demo data. What kind of motion do these particles exhibit?';
  error(null);
  return {...sample,questionStaged:stageQuestion};
}
$('sample').onclick = () => loadSample().catch(error);
$('new').onclick = () => {
  clearTimeout(pollTimer); active = null; latest = null; lastSignature = '';started=false;pendingFiles=[];sidebar(false);
  creationKey = crypto.randomUUID() + crypto.randomUUID();
  resetEmbedding();
  setFiles([]); $('messages').replaceChildren(); $('system-activity').replaceChildren();
  $('status').textContent = 'Ready when you are'; $('question').value = '';
  $('files').disabled = $('sample').disabled = false;
  $('downloads').replaceChildren(); $('cancel').hidden = true;
  $('result-count').textContent='No outputs yet';$('results-panel').open=false;
  history.replaceState(null, '', './#new=1'); renderHistory(); error(null);showJourney();
};
async function submitQuestion(question) {
  if (sending) throw new Error('A request is already being sent.');
  if (!question.trim()) throw new Error('Enter your question first.');

  sending = true; $('send').disabled = true;renderFileControls();
  try {
    let q = question.trim();
    const demoRequest=q.match(/\b(?:use|try|load|show|give)\b(?:(?!\b(?:no|without)\b)[^.!?]){0,50}\b(?:demo|sample)\b|^(?:demo|sample)(?: data)?[.!?]?$/i);
    const demoNegated=demoRequest&&/\b(?:don't|do not|not|never|without)\s+(?:\w+\s+){0,2}$/i.test(q.slice(0,demoRequest.index));
    if(!files.length&&demoRequest&&!demoNegated){
      await loadSample(false);
    }
    if(sampleFile&&files.includes(sampleFile)&&(!active||pendingFiles.includes(sampleFile)))q+='\n\nDemo dataset context:\n'+sampleContext;
    started=true;showJourney();$('status').textContent='Sending your question…';
    const selection=selected.size?{ids:[...selected],totalCount:selected.size}:null;
    if (!active) {
      active = await api('/api/jobs', {files, question: q, creationKey,selection}, null);
      active.title=question.trim().slice(0,70);remember(active); history.replaceState(null, '', './#job=' + active.id);
    } else {
      if (!pendingFollowup || pendingFollowup.job !== active.id || pendingFollowup.question !== q||JSON.stringify(pendingFollowup.selection)!==JSON.stringify(selection))
        pendingFollowup = {job: active.id, question: q, selection,requestId: crypto.randomUUID()};
      await api(`/api/jobs/${active.id}/messages`, {question: q, selection:pendingFollowup.selection,requestId: pendingFollowup.requestId,...(pendingFiles.length?{files:pendingFiles}:{})});
      pendingFollowup = null;pendingFiles=[];
    }
    $('question').value = ''; error(null); await refresh();
    return {id: active.id, status: latest?.status};
  } finally { if(!active)started=false;showJourney();sending = false; $('send').disabled = false;renderFileControls(); }
}
$('composer').onsubmit = event => { event.preventDefault(); submitQuestion($('question').value).catch(error); };
$('cancel').onclick = async () => {
  try { await api(`/api/jobs/${active.id}/cancel`, {}); await refresh(); } catch(err) { error(err); }
};
function message(author, text, turn=Infinity) {
  const block = document.createElement('div'); block.className = 'message ' + (author === 'SPT AGENT' ? 'agent' : 'user');
  const label = document.createElement('span'); label.className = 'author'; label.textContent = author;
  block.append(label,author==='SPT AGENT'?renderAnswer(text,href=>resolveArtifact(href,turn),openArtifact):document.createTextNode(text)); return block;
}
function render(job) {
  latest = job;showJourney();
  const status = {queued:'Queued · waiting for the desktop', running:'Analyzing your dataset',
    completed:'Analysis complete',failed:'Analysis needs attention',cancelled:'Analysis stopped',
    cancelling:'Stopping analysis',interrupted:'Desktop connection interrupted'};
  $('status').textContent = status[job.status] || job.status;
  $('cancel').hidden = !['queued', 'running', 'cancelling'].includes(job.status);
  renderFileControls();
  $('send-hint').textContent = ['running','queued'].includes(job.status) ? 'Follow-ups will run after the current analysis.' : 'You can return to this conversation from this browser.';
  if (!files.length&&job.files.length) setFiles(job.files);
  const currentTurn=job.turns.findIndex(t=>t.status==='running');
  const projectionStage=job.events.filter(e=>e.kind==='progress'&&e.turn===currentTurn).at(-1)?.stage;
  const waiting=!embedding&&!points.length&&(['running','queued'].includes(job.status)||projectionStage==='embeddings_unavailable')&&job.files.length>0;
  $('projection-progress').hidden=!waiting;
  $('projection-progress-text').textContent=projectionStage==='embeddings_unavailable'?'Projection preparation or transfer failed. The agent can preprocess the data and retry publishing the embeddings.':projectionStage==='embeddings'?'Computing and transferring encoder vectors…':'Preparing the projection. Other file formats may need the agent to normalize columns and units first.';
  const vectorFiles=job.artifacts.filter(f=>f.name.startsWith('encoder_embeddings')&&f.name.endsWith('.json'));
  const vectorVersion=vectorFiles.map(f=>f.id).join(':');
  if(vectorFiles.length&&vectorVersion!==embeddingArtifactsVersion){
    try{
      const candidate=readEmbeddings(vectorFiles,decode);
      const rawIds=new Set(points.filter(p=>!p.fromEncoder).map(p=>p.key)),rawFiles=new Set(points.filter(p=>!p.fromEncoder).map(p=>p.file));
      candidate.points=candidate.points.filter(p=>candidate.authoritativeSources.includes(p.file)||!rawFiles.has(p.file)||rawIds.has(p.id));
      if(candidate.points.length&&(!embedding||candidate.version!==embedding.version)){
        embedding=candidate;
        const normalizedSources=new Set(candidate.points.filter(p=>p.xPosition?.length&&p.xPosition.length===p.yPosition?.length).map(p=>p.file));
        points=points.filter(p=>!p.fromEncoder&&!normalizedSources.has(p.file));
        const ids=new Set(points.map(p=>p.key));
        for(const p of candidate.points)if(!ids.has(p.id)&&p.xPosition?.length===p.yPosition?.length&&p.xPosition?.length){points.push({id:p.trackId,key:p.id,file:p.file,path:p.xPosition.map((x,i)=>[x,p.yPosition[i]]),fromEncoder:true});ids.add(p.id);}
        for(const option of $('projection').options)option.disabled=false;
        $('embedding-info').textContent=candidate.points.length+' × '+candidate.points[0].vector.length+' encoder vectors';
        $('projection-progress').hidden=true;
        legacyVersion=null;autoViewVersion=candidate.version;embeddingId=candidate.version;
        setProjection('umap').catch(error);
      }
      embeddingArtifactsVersion=vectorVersion;
    }catch(err){error(err);}
  }
  const nextColorVersion=JSON.stringify(job.pointColoring||null);
  if(nextColorVersion!==colorVersion){try{setColoring(job.pointColoring);colorVersion=nextColorVersion;}catch(err){error(err);}}
  const signature = JSON.stringify([job.updated, job.events.length, job.status]);
  if (signature === lastSignature) return;
  lastSignature = signature;
  const nearBottom = $('latest-message').hidden;
  const expanded=new Set([...$('messages').querySelectorAll('details[open][data-step]')].map(e=>e.dataset.step));
  const progressEvents=job.events.filter(e=>e.kind==='summary'&&e.summary)
    .sort((a,b)=>a.turn-b.turn||a.order-b.order||a.id-b.id);
  const blocks=[];
  for(const [turn,t] of job.turns.entries()){
    const display=questionForDisplay(t),userBubble=message(t.requestId?.startsWith('operator-recovery-')?'RECOVERY':'YOU',display.text);
    if(display.selection){const badge=document.createElement('small'),s=display.selection;badge.className='selection-context';badge.textContent=s.legacy?'Trajectory selection attached':s.totalCount>s.ids.length?`First ${s.ids.length} of ${s.totalCount} selected trajectories attached`:`${s.ids.length} selected trajector${s.ids.length===1?'y':'ies'} attached`;userBubble.append(badge);}
    blocks.push(userBubble);
    for(const e of progressEvents.filter(e=>e.turn===turn)){
      const bubble=document.createElement('details');
      bubble.className='message agent progress-summary';
      bubble.dataset.step=job.id+':'+e.id;
      bubble.open=expanded.has(bubble.dataset.step);
      const heading=document.createElement('summary');
      const title=document.createElement('span');title.className='step-label';
      title.textContent='Analysis step '+(progressEvents.indexOf(e)+1)+(e.retrospective?' · Retrospective':'');
      const preview=document.createElement('span');preview.className='step-preview';
      preview.textContent=e.summary.checked;
      heading.append(title,preview);bubble.append(heading);
      const labels=e.summary.phase==='plan'?[['checked','Approach'],['observed','Available now'],['next','Next']]:[['checked','Checked'],['observed','Observed'],['next','Next']];
      for(const [key,label] of labels){
        if(!e.summary[key])continue;
        const p=document.createElement('p'),strong=document.createElement('strong');
        strong.textContent=label+': ';p.append(strong,document.createTextNode(e.summary[key]));bubble.append(p);
      }
      blocks.push(bubble);
    }
    if(t.answer)blocks.push(message('SPT AGENT',t.answer,turn));
    else if(t.status==='queued')blocks.push(message('SPT AGENT','Your question is queued.'));
    else if(t.status==='running'){
      const update=job.events.filter(e=>e.kind==='public_update'&&e.turn===turn).at(-1);
      const bubble=message('SPT AGENT',update?.text||'Analyzing your dataset…',turn);
      if(update){const tag=document.createElement('small');tag.className='live-answer-note';tag.textContent='Live update · analysis in progress';bubble.prepend(tag);}
      blocks.push(bubble);
    }
  }
  $('messages').replaceChildren(...blocks);
  if(nearBottom)requestAnimationFrame(focusLatest);else $('latest-message').hidden=false;
  $('system-activity').replaceChildren(...job.events.filter(e=>!['summary','message','tool','public_update'].includes(e.kind)).map(e=>{
    const line=document.createElement('p');line.textContent=new Date(e.at*1000).toLocaleTimeString()+' · '+e.text;return line;
  }));
  $('downloads').replaceChildren();
  const displayedArtifacts = new Set();
  for (const file of job.artifacts) {
    if (file.name.startsWith('mplconfig__')) continue;
    const artifactKey = JSON.stringify([file.name, file.sha256 || file.data]);
    if(displayedArtifacts.has(artifactKey))continue;displayedArtifacts.add(artifactKey);
    const a=document.createElement('button');a.className='artifact';a.textContent=file.name;a.onclick=()=>openArtifact(file);$('downloads').append(a);
  }
  $('result-count').textContent=displayedArtifacts.size?displayedArtifacts.size+' file'+(displayedArtifacts.size===1?'':'s'):'No outputs yet';
}
async function refresh() {
  clearTimeout(pollTimer);
  if (!active) return;
  const target = active;
  try {
    const job = await api('/api/jobs/' + target.id, undefined, target);
    if (active?.id === target.id) {
      if(connectionError&&$('error').textContent===connectionError)error(null);
      render(job);
      projectorBridge.tick(job,target).catch(()=>{}); // Cached result retries on next poll.
    }
  } catch(err) {
    if(active?.id===target.id){
      const message='Connection interrupted. Retrying; the analysis can continue on the desktop. '+err.message;
      error(new Error(message));connectionError=message;
    }
  }
  finally { if (active) pollTimer = setTimeout(refresh, cloudPollDelay(latest,document.hidden)); }
}

// Spatial preview is explicitly raw trajectories, never a fabricated embedding.
function preview(keepSelection=false) {
  points=[];if(!keepSelection)selected.clear();
  for (const file of files) {
    if (!/\.(csv|tsv)$/i.test(file.name)) continue;
    // Large datasets are preprocessed by the agent; avoid splitting millions of rows on the UI thread.
    if ((file.bytes??file.data.length*3/4)>16*1024*1024) continue;
    const lines=decode(file.data).trim().split(/\r?\n/); const sep=file.name.endsWith('.tsv')?'\t':',';
    const columns=lines[0].replace(/^\uFEFF/,'').split(sep).map(s=>s.trim().replace(/^"|"$/g,''));
    const ix=columns.indexOf('x_um'),iy=columns.indexOf('y_um'),it=columns.indexOf('track_id');
    if(ix<0||iy<0||it<0)continue;
    const tracks=new Map();
    for(const line of lines.slice(1)){
      const row=line.split(sep); const x=Number(row[ix]),y=Number(row[iy]);
      if(!Number.isFinite(x)||!Number.isFinite(y))continue;
      const id=row[it]?.trim();if(!id)continue;
      const key=file.name+':'+id;
      if(!tracks.has(key))tracks.set(key,{id, key, file:file.name, path:[]});
      tracks.get(key).path.push([x,y]);
    }
    points.push(...tracks.values());
  }
  $('plot-empty').hidden=points.length>0;
  $('plot-empty').textContent=files.length?'No automatic spatial preview for this file format. The agent can still inspect and preprocess it.':'Upload trajectories to inspect their paths. You can ask a question without selecting points.';
  $('plot-meta').textContent=points.length?`${points.length} trajectories · x/y in µm`:'No spatial preview';
  draw();
}
const canvas=$('plot');let projected=[];
function updatePlotHelp(method=$('projection').value){
  $('plot-help').textContent=method==='raw'?'Click a trajectory to inspect it; drag a lasso to select several. Selected tracks accompany your next question.':'Each point represents one trajectory’s encoder vector. Projection distances have no spatial units and clusters alone do not establish a motion model. Click a point to inspect its trajectory.';
}
function draw(){
  updatePlotHelp();
  const box=canvas.getBoundingClientRect();const scale=devicePixelRatio||1;
  canvas.width=box.width*scale;canvas.height=box.height*scale;
  const c=canvas.getContext('2d');c.scale(scale,scale);c.clearRect(0,0,box.width,box.height);
  c.strokeStyle='#e6eaf0';c.lineWidth=.5;
  for(let i=1;i<8;i++){c.beginPath();c.moveTo(i*box.width/8,0);c.lineTo(i*box.width/8,box.height);c.stroke();c.beginPath();c.moveTo(0,i*box.height/8);c.lineTo(box.width,i*box.height/8);c.stroke();}
  const visible=projectedEmbedding ? projectedEmbedding.map(p=>({id:p.id,key:p.id,path:[[p.x,p.y]]})) : points;
  if(!visible.length)return;
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
  for(const t of visible)for(const [x,y]of t.path){minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);}
  const s=Math.min((box.width-48)/(maxX-minX||1),(box.height-48)/(maxY-minY||1));
  projected=visible.map((t,i)=>{
    const path=t.path.map(([x,y])=>[24+(x-minX)*s,box.height-24-(y-minY)*s]);
    c.strokeStyle=selected.size?(selected.has(t.key)?'#d6584f':'#9ca9be'):(pointColoring?(pointColoring.colors.get(t.key)||'#b7bec8'):['#547cce','#388c94','#9671b2','#b18745'][i%4]);
    c.globalAlpha=selected.size&&!selected.has(t.key)?.25:.8;c.lineWidth=selected.has(t.key)?1.6:.9;c.beginPath();
    path.forEach(([x,y],j)=>j?c.lineTo(x,y):c.moveTo(x,y));c.stroke();
    if(projectedEmbedding){c.fillStyle=c.strokeStyle;c.beginPath();c.arc(path[0][0],path[0][1],selected.has(t.key)?4:3,0,Math.PI*2);c.fill();}
    return {...t,path};
  });c.globalAlpha=1;
  if(polygon.length){c.beginPath();polygon.forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.strokeStyle='#4779dc';c.lineWidth=1.5;c.stroke();}
  $('selection').textContent=selected.size?`${selected.size} selected trajector${selected.size===1?'y':'ies'}`:'All trajectories';
}
function inside([x,y],poly){let hit=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){const [xi,yi]=poly[i],[xj,yj]=poly[j];if((yi>y)!=(yj>y)&&x<(xj-xi)*(y-yi)/(yj-yi)+xi)hit=!hit;}return hit;}
canvas.onpointerdown=e=>{drawing=true;polygon=[[e.offsetX,e.offsetY]];canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(drawing){polygon.push([e.offsetX,e.offsetY]);draw();}};
canvas.onpointerup=e=>{
  if(drawing){
    const start=polygon[0],end=[e.offsetX,e.offsetY];
    if(polygon.every(p=>Math.hypot(p[0]-start[0],p[1]-start[1])<=5)&&Math.hypot(end[0]-start[0],end[1]-start[1])<=5){
      const hit=pickTrajectory(projected,end);selected=new Set(hit?[hit]:[]);
    }else if(polygon.length>2)selected=new Set(projected.filter(t=>t.path.some(p=>inside(p,polygon))).map(t=>t.key));
  }
  drawing=false;polygon=[];draw();showTrajectory([...selected]);
};
canvas.onpointercancel=()=>{drawing=false;polygon=[];draw();};
$('clear-selection').onclick=()=>{selectTracks([]);showTrajectory([]);};
$('projection-settings').onclick=()=>{$('view-panel').open=false;legacyApi?.openParameters();};
new ResizeObserver(draw).observe(canvas);
function panels(data,chat){$('explorer').hidden=!data;$('conversation').hidden=!chat;$('show-data').hidden=data;$('show-chat').hidden=chat;main.classList.toggle('data-hidden',!data);main.classList.toggle('chat-hidden',!chat);requestAnimationFrame(draw);}
$('hide-data').onclick=()=>panels(false,true);$('show-data').onclick=()=>panels(true,true);$('hide-chat').onclick=()=>panels(true,false);$('show-chat').onclick=()=>panels(true,true);

if(document.modelContext?.registerTool){
  const lifecycle=new AbortController();
  const register=t=>Promise.resolve(document.modelContext.registerTool(t,{signal:lifecycle.signal})).catch(()=>{});
  register({name:'read_spt_workspace',description:'Read current dataset, job status, and selected track IDs.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true},execute:()=>({files:files.map(f=>f.name),job:active?.id,status:latest?.status,selectedTrackIds:[...selected]})});
  register({name:'stage_spt_sample',description:'Load the sample dataset and stage its question without starting analysis.',inputSchema:{type:'object',properties:{},additionalProperties:false},execute:loadSample});
  register({name:'set_spt_projection',description:'Compute and show PCA, UMAP, or t-SNE from the available encoder vectors, or show raw trajectories.',inputSchema:{type:'object',properties:{method:{type:'string',enum:['raw','pca','umap','tsne']}},required:['method'],additionalProperties:false},execute:input=>setProjection(input?.method)});
  register({name:'select_spt_tracks',description:'Select file-qualified track IDs (filename:track_id) in the visible trajectory preview.',inputSchema:{type:'object',properties:{ids:{type:'array',items:{type:'string'}}},required:['ids'],additionalProperties:false},execute:input=>{if(!input||!Array.isArray(input.ids)||input.ids.some(id=>typeof id!=='string'||!points.some(t=>t.key===id)&&!embedding?.points.some(p=>p.id===id)))throw new Error('Unknown track ID');selected=new Set(input.ids);draw();return {selectedTrackIds:[...selected]};}});
  window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
}
function resolveArtifact(href,turn=Infinity){return resolveArtifactLink(href,latest?.artifacts||[],turn);}
function openArtifact(file){
  artifactUrls.forEach(url=>URL.revokeObjectURL(url));artifactUrls=[];
  const bytes=Uint8Array.from(atob(file.data),c=>c.charCodeAt(0)),ext=file.name.split('.').at(-1).toLowerCase();
  const type=({png:'image/png',jpg:'image/jpeg',pdf:'application/pdf',csv:'text/csv'})[ext]||'text/plain';
  const url=URL.createObjectURL(new Blob([bytes],{type}));artifactUrls.push(url);
  $('artifact-title').textContent=file.name;$('artifact-download').href=url;$('artifact-download').download=file.name;
  const preview=$('artifact-preview');preview.replaceChildren();
  if(['png','jpg'].includes(ext)){const img=document.createElement('img');img.src=url;img.alt=file.name;preview.append(img);}
  else if(ext==='pdf'){const p=document.createElement('p');p.textContent='Download this PDF to open it in your document viewer.';preview.append(p);}
  else {const text=new TextDecoder().decode(bytes);if(ext==='md')preview.append(renderAnswer(text,href=>resolveArtifact(href,file.turn),openArtifact));else{const pre=document.createElement('pre');pre.textContent=text.slice(0,200000)+(text.length>200000?'\n… Download the full file to read more.':'');preview.append(pre);}}
  if(!$('artifact-dialog').open)$('artifact-dialog').showModal();
}
$('artifact-close').onclick=()=>$('artifact-dialog').close();
$('artifact-dialog').onclick=e=>{if(e.target===$('artifact-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}};
function focusLatest(){
  const log=$('messages'),last=log.lastElementChild;if(!last)return;
  // Put the beginning of the newest response near the reading center, with room below.
  log.scrollTop=last.offsetTop-log.offsetTop-log.clientHeight*.25;$('latest-message').hidden=true;
}
$('latest-message').onclick=focusLatest;
$('messages').addEventListener('scroll',()=>{const last=$('messages').lastElementChild;if(!last)return;const r=last.getBoundingClientRect(),box=$('messages').getBoundingClientRect();$('latest-message').hidden=r.top<box.bottom&&r.bottom>box.top;},{passive:true});
let trajectoryFrame=0,trajectoryRunning=true,trajectoryTimer=null,trajectoryPath=[];
function showTrajectory(ids){
  if(ids.length!==1){$('trajectory-card').hidden=true;selectedCardId=null;clearTimeout(trajectoryTimer);return;}
  const track=points.find(t=>t.key===ids[0]);if(!track?.path.length)return;
  if(selectedCardId===track.key)return;selectedCardId=track.key;trajectoryPath=track.path;trajectoryFrame=0;trajectoryRunning=true;
  $('trajectory-title').textContent=track.key;$('trajectory-card').hidden=false;$('trajectory-play').textContent='Pause';
  clearTimeout(trajectoryTimer);animateTrajectory();
}
function animateTrajectory(){
  if($('trajectory-card').hidden)return;
  const c=$('trajectory-animation'),ctx=c.getContext('2d'),w=260,h=170,scale=devicePixelRatio||1;c.width=w*scale;c.height=h*scale;ctx.scale(scale,scale);
  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  for(const [x,y] of trajectoryPath){minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);}
  const fit=Math.min((w-28)/(maxX-minX||1),(h-28)/(maxY-minY||1)),ox=(w-(maxX-minX)*fit)/2,oy=(h-(maxY-minY)*fit)/2;
  const coords=trajectoryPath.map(([x,y])=>[ox+(x-minX)*fit,h-oy-(y-minY)*fit]);
  const line=(end,color)=>{ctx.beginPath();coords.slice(0,end).forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.strokeStyle=color;ctx.lineWidth=1.4;ctx.stroke();};
  line(coords.length,'#e0e6ef');line(trajectoryFrame+1,'#4779dc');const [x,y]=coords[trajectoryFrame];ctx.beginPath();ctx.arc(x,y,3.5,0,Math.PI*2);ctx.fillStyle='#d6584f';ctx.fill();
  $('trajectory-frame').textContent=`Frame ${trajectoryFrame+1} / ${coords.length}`;
  if(trajectoryRunning){trajectoryFrame=(trajectoryFrame+1)%coords.length;trajectoryTimer=setTimeout(animateTrajectory,Math.max(16,4000/coords.length));}
}
$('close-trajectory').onclick=()=>{$('trajectory-card').hidden=true;clearTimeout(trajectoryTimer);};
$('trajectory-play').onclick=()=>{trajectoryRunning=!trajectoryRunning;$('trajectory-play').textContent=trajectoryRunning?'Pause':'Play';clearTimeout(trajectoryTimer);if(trajectoryRunning)animateTrajectory();};
renderHistory();showJourney();
if(active)refresh();else {
  draw();
  if(fragment.has('job'))error(new Error('This analysis is not available in this browser. Return to the browser where you started it, or start a new analysis here.'));
}

document.addEventListener('visibilitychange',()=>{if(!document.hidden&&active)refresh();});
