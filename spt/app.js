import {SPT_API_ORIGIN} from "./deployment-config.js?v=15494e176060da07";
import {createCloudApi,cloudPollDelay} from "./cloud-client.js?v=15494e176060da07";
import {renderAnswer,resolveArtifactLink} from './answer-renderer.js?v=15494e176060da07';
import {artifactVersion} from './artifact-links.js?v=15494e176060da07';
import {buildAnalysisBundle,listedResults} from './result-bundle.js?v=15494e176060da07';
import {parseTrajectoryTable,matchingTracks,rawPreviewTracks,encoderPreviewTrack} from './trajectory-data.js?v=15494e176060da07';
import {ProjectorClient, readEmbeddings, questionForDisplay, pickTrajectory, selectSavedJob, planProjectionSelection} from './projector-client.js?v=15494e176060da07';
import {createColoring} from './coloring.js?v=15494e176060da07';
import {projectionFigureContext,renderProjectionFigure} from './projection-figure.js?v=15494e176060da07';
import {createViewRecovery} from './view-recovery.js?v=15494e176060da07';
const MAX_UPLOAD_BYTES=128*1024*1024;
const $ = id => document.getElementById(id);
const main = document.querySelector('main');
let files = [], active = null, latest = null, points = [], selected = new Set(), polygon = [], drawing = false;
let artifactUrls = [], pendingFiles=[], started=false, autoViewVersion=null, selectedCardId=null;
let bundleUrl=null,bundleBusy=false,figureError=null;
let embedding = null, embeddingId = null, projectionWorker = null, projectedEmbedding = null;
let embeddingArtifactsVersion = null;
let finishProjection = null;
let legacyApi=null, legacyVersion=null, legacyMode=false, projectionGeneration=0;
let pendingFollowup = null,pendingNativeSelection=null;
let pointColoring=null,colorVersion='null';
let projectionViewRevision=null,projectionViewIds=null,viewSelectionError=null,rawSelectionError=null;
let pollTimer, sending = false, lastSignature = '', creationKey = crypto.randomUUID() + crypto.randomUUID();
let sendingWorkspaceKey=null,lastBrowserAddress='';
let sampleFile=null, sampleContext='';
// Keep file references in this tab, without copying uploads into browser storage.
const workspaceDrafts=new Map();
const viewRecovery=createViewRecovery({getItem:key=>sessionStorage.getItem(key),setItem:(key,value)=>sessionStorage.setItem(key,value),removeItem:key=>sessionStorage.removeItem(key)});
let viewRecoveryFailed=false;
let draftToRestore=null,loadingWorkspace=false,unavailableWorkspace=false,recoveryDraftKey=null,missingRecoveryFiles=false,missingAnalysisLink=false;
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
missingAnalysisLink=fragment.has('job')&&!active;

function remember(job) {
  const saved=savedJobs(),previous=saved.find(j=>j.id===job.id);
  const entry={...previous,...job,title:job.title||previous?.title||'Analysis'};
  if(saved.at(-1)?.id===job.id&&JSON.stringify(previous)===JSON.stringify(entry))return;
  const all=saved.filter(j=>j.id!==job.id);all.push(entry);
  localStorage.setItem('spt.jobs', JSON.stringify(all));
  renderHistory();
}
function analysisLabel(job){return (job.title||'Analysis')+' · '+job.id.slice(-8);}
function renderHistory() {
  const options = [Object.assign(document.createElement('option'), {value:'', textContent:'Choose an analysis'})];
  for (const job of savedJobs().reverse()) options.push(Object.assign(document.createElement('option'), {value:job.id, textContent:analysisLabel(job)}));
  $('history').replaceChildren(...options); $('history').value = active?.id || '';
  $('conversation-list').replaceChildren(...savedJobs().reverse().map(job=>{const b=document.createElement('button');b.textContent=analysisLabel(job);b.title=(job.title||'Analysis')+' · '+job.id;b.className=active?.id===job.id?'current':'';b.disabled=sending;b.onclick=()=>{$('history').value=job.id;sidebar(false);$('history').onchange();};return b;}));
  renderDraftHistory();
}
function workspaceKey(){return active?.id||'new:'+creationKey;}
function browserAddress(){return location.href+'\n'+(history.state?.sptWorkspace||'');}
function writeWorkspaceAddress(mode='push'){
  const url=active?'./#job='+active.id:'./#new=1';
  history[mode==='push'?'pushState':'replaceState']({sptWorkspace:workspaceKey()},'',url);
  lastBrowserAddress=browserAddress();
}
function saveWorkspaceDraft(){
  const key=unavailableWorkspace?recoveryDraftKey:workspaceKey(),question=$('question').value;
  if(!active&&!question.trim()&&!files.length){workspaceDrafts.delete(key);return;}
  workspaceDrafts.set(key,{question,selection:loadingWorkspace?(draftToRestore?.selection||[...selected]):[...selected],projection:loadingWorkspace?(draftToRestore?.projection||$('projection').value):$('projection').value,
    projectionSettings:loadingWorkspace?draftToRestore?.projectionSettings:legacyApi?.viewSettings?.()||workspaceDrafts.get(key)?.projectionSettings,
    ...(!active||unavailableWorkspace?{files,sampleFile:sampleFile||(unavailableWorkspace&&storedDemoContext()?files[0]:null),sampleContext:sampleContext||(unavailableWorkspace?storedDemoContext():''),creationKey:unavailableWorkspace?recoveryDraftKey.slice(4):creationKey,missingFiles:unavailableWorkspace?!files.length:missingRecoveryFiles||(missingAnalysisLink&&!files.length&&!!question.trim())}:{})});
  if(active&&!unavailableWorkspace&&!loadingWorkspace&&!pendingNativeSelection)viewRecoveryFailed=!viewRecovery.save(active.id,workspaceDrafts.get(key));
}
window.addEventListener('pagehide',saveWorkspaceDraft);
function renderDraftHistory(){
  const drafts=[...workspaceDrafts].filter(([key])=>key.startsWith('new:')&&key!==workspaceKey());
  $('draft-section').hidden=!drafts.length;
  $('draft-list').replaceChildren(...drafts.map(([key,draft])=>{
    const button=document.createElement('button');button.disabled=sending;
    button.textContent=draft.question.trim().slice(0,70)||draft.files.map(f=>f.name).join(', ')||'Recovered analysis draft';
    button.onclick=()=>{if(sending)return;saveWorkspaceDraft();sidebar(false);openUnsentWorkspace(workspaceDrafts.get(key)||draft);};return button;
  }));
}
function storedDemoContext(){return latest?.turns?.find(t=>t.question.includes('\n\nDemo dataset context:\n'))?.question.split('\n\nDemo dataset context:\n')[1]||'';}
function resetUnavailable(){unavailableWorkspace=false;recoveryDraftKey=null;missingAnalysisLink=false;$('expired-session').hidden=true;}
function expireWorkspace(id){makeWorkspaceUnavailable(id,'expired');}
function handleAnalysisFailure(path,err,auth){
  if(!auth?.id)return;
  const jobPath='/api/jobs/'+auth.id;
  const missingSession=err.status===404&&err.resource!=='object'&&[jobPath,jobPath+'/messages',jobPath+'/prepare',jobPath+'/cancel'].includes(path);
  if(err.status===401||err.status===403||missingSession)makeWorkspaceUnavailable(auth.id,missingSession?'missing':'access');
}
function makeWorkspaceUnavailable(id,reason){
  if(!active||id!==active.id||unavailableWorkspace)return;
  unavailableWorkspace=true;recoveryDraftKey||='new:'+crypto.randomUUID()+crypto.randomUUID();
  // Keep local work while pausing reads that require access recovery.
  saveWorkspaceDraft();clearTimeout(pollTimer);loadingWorkspace=false;
  if(connectionError&&$('error').textContent===connectionError)error(null);connectionError=null;
  const expired=reason==='expired';
  $('status').textContent=expired?'Analysis expired':'Analysis unavailable';$('cancel').hidden=true;$('expired-session').hidden=false;
  const explanation=expired?'This analysis expired after seven days.':'This analysis cannot be opened with the access saved in this browser. Try again, or return to the browser or link where you started it.';
  $('expiry-explanation').textContent=explanation+(files.length?' The dataset and question currently in this tab can continue in a new analysis. Results already loaded here can still be downloaded.':' Continue with your current question, then re-upload the original files to analyze that dataset.');
  $('continue-analysis').textContent=files.length?'Continue with this dataset':'Continue with your question';
  $('retry-analysis').hidden=expired;
  $('send-hint').textContent='This unavailable analysis cannot accept follow-ups.';renderHistory();renderFileControls();
}
$('retry-analysis').onclick=async()=>{
  if(sending||!active||!unavailableWorkspace)return;
  saveWorkspaceDraft();unavailableWorkspace=false;loadingWorkspace=true;$('expired-session').hidden=true;error(null);
  $('status').textContent='Loading analysis…';renderFileControls();await refresh();
};
$('start-new-analysis').onclick=()=>{
  if(sending||!missingAnalysisLink)return;
  saveWorkspaceDraft();openUnsentWorkspace(workspaceDrafts.get(workspaceKey()));$('question').focus();
};
$('continue-analysis').onclick=()=>{if(sending||!unavailableWorkspace)return;saveWorkspaceDraft();const draft=workspaceDrafts.get(recoveryDraftKey);openUnsentWorkspace(draft);$('question').focus();};
function clearConversation(){
  if($('artifact-dialog').open)$('artifact-dialog').close();
  $('artifact-preview').replaceChildren();artifactUrls.forEach(url=>URL.revokeObjectURL(url));artifactUrls=[];
  $('messages').replaceChildren();$('system-activity').replaceChildren();$('downloads').replaceChildren();
  $('bundle-controls').hidden=true;$('bundle-status').textContent='';
  $('status').textContent='Ready when you are';$('cancel').hidden=true;$('latest-message').hidden=true;
  $('result-count').textContent='No outputs yet';$('results-panel').open=false;
}
function restoreDraftSelection(draft){
  selected=new Set(draft?.selection||[]);draw();syncTrackChoice();showTrajectory([...selected]);
}
function openUnsentWorkspace(draft=null,{historyMode='push'}={}){
  viewRecoveryFailed=false;
  clearTimeout(pollTimer);active=null;latest=null;lastSignature='';started=!!draft?.files?.length;pendingFiles=[];draftToRestore=null;loadingWorkspace=false;
  resetUnavailable();missingRecoveryFiles=!!draft?.missingFiles;
  creationKey=draft?.creationKey||crypto.randomUUID()+crypto.randomUUID();
  sampleFile=draft?.sampleFile||null;sampleContext=draft?.sampleContext||'';
  if(sampleContext)$('demo-context').replaceChildren(renderAnswer(sampleContext));
  resetEmbedding();clearConversation();setFiles(draft?.files||[]);$('question').value=draft?.question||'';
  restoreDraftSelection(draft?.missingFiles?{...draft,selection:[]}:draft);if(historyMode)writeWorkspaceAddress(historyMode);renderHistory();error(null);showJourney();
}
async function openSavedWorkspace(job,{historyMode='push'}={}){
  if(!job||job.id===active?.id)return;
  saveWorkspaceDraft();resetUnavailable();missingRecoveryFiles=false;clearTimeout(pollTimer);active=job;error(null);latest=null;lastSignature='';pendingFiles=[];started=false;
  viewRecoveryFailed=false;
  sampleFile=null;sampleContext='';draftToRestore=workspaceDrafts.get(job.id)||viewRecovery.load(job.id);loadingWorkspace=true;
  if(draftToRestore)workspaceDrafts.set(job.id,draftToRestore);
  $('question').value=draftToRestore?.question||'';resetEmbedding();clearConversation();setFiles([]);$('status').textContent='Loading analysis…';renderHistory();showJourney();
  if(historyMode)writeWorkspaceAddress(historyMode);await refresh();
}
$('history').onchange=async()=>{
  if(sending)return;
  await openSavedWorkspace(savedJobs().find(j=>j.id===$('history').value));
};
async function restoreBrowserWorkspace(){
  const address=browserAddress();if(address===lastBrowserAddress)return;
  lastBrowserAddress=address;
  const fragment=new URLSearchParams(location.hash.slice(1)),draftKey=history.state?.sptWorkspace;
  let job=selectSavedJob(savedJobs(),fragment);
  if(fragment.has('job')&&fragment.has('key')){
    job={id:fragment.get('job'),token:fragment.get('key')};remember(job);
  }
  // A draft submitted while the user navigated away becomes its saved analysis.
  if(!fragment.has('job')&&typeof draftKey==='string')job=savedJobs().find(j=>j.sourceDraftKey===draftKey)||job;
  if(job){
    await openSavedWorkspace(job,{historyMode:'replace'});
    if(active?.id===job.id)writeWorkspaceAddress('replace');
  }else{
    if(!fragment.has('job')&&draftKey===workspaceKey()&&!missingAnalysisLink)return;
    saveWorkspaceDraft();
    const draft=workspaceDrafts.get(draftKey)||(typeof draftKey==='string'&&draftKey.startsWith('new:')?{creationKey:draftKey.slice(4)}:null);
    openUnsentWorkspace(draft,{historyMode:null});
    missingAnalysisLink=fragment.has('job');
    if(missingAnalysisLink){history.replaceState({sptWorkspace:workspaceKey()},'',location.href);lastBrowserAddress=browserAddress();renderFileControls();showJourney();}
    else writeWorkspaceAddress('replace');
  }
}
window.addEventListener('popstate',()=>restoreBrowserWorkspace().catch(error));
window.addEventListener('hashchange',()=>restoreBrowserWorkspace().catch(error));
function sidebar(open){
  const panel=$('session-sidebar'),wasOpen=!panel.hidden,hadFocus=panel.contains(document.activeElement);
  panel.hidden=!open;$('toggle-sidebar').setAttribute('aria-expanded',String(open));
  if(open)$('close-sidebar').focus();else if(wasOpen&&hadFocus)$('toggle-sidebar').focus();
}
$('toggle-sidebar').onclick=()=>sidebar($('session-sidebar').hidden);$('close-sidebar').onclick=()=>sidebar(false);
$('session-sidebar').onkeydown=event=>{if(event.key==='Escape'){event.preventDefault();event.stopPropagation();sidebar(false);}};
function showJourney(){
  const talking=!!active||started;main.classList.toggle('missing-link',missingAnalysisLink);$('missing-analysis').hidden=!missingAnalysisLink;main.classList.toggle('onboarding',!talking);$('welcome').hidden=talking;
  const destination=talking?$('data-panel'):$('welcome-upload');
  if($('upload-box').parentElement!==destination)destination.append($('upload-box'));
  if(!talking)panels(true,true);
  if(!active&&!sending)$('send-hint').textContent=files.length?'Unsent analysis. Your dataset will accompany your question.':'Include position units and time between frames, or try demo data.';
  if(missingAnalysisLink){$('status').textContent='Linked analysis unavailable';$('send-hint').textContent='Start a new analysis here to send this question.';}
  renderSendHint();
}
const cloudApi=createCloudApi({origin:SPT_API_ORIGIN,encode,onUploadProgress:(sent,total)=>{if(sendingWorkspaceKey===workspaceKey())$('status').textContent='Uploading '+Math.round(sent/Math.max(1,total)*100)+'%…';},onExpiry:id=>{
  const all=savedJobs().filter(j=>j.id!==id);localStorage.setItem('spt.jobs',JSON.stringify(all));expireWorkspace(id);renderHistory();
}});
async function api(path,data,auth=active){
  let result;
  try{result=await cloudApi(path,data,auth);}catch(err){handleAnalysisFailure(path,err,auth);throw err;}
  if(result.id===active?.id&&!unavailableWorkspace&&result.expiresAt)active.expiresAt=result.expiresAt;
  return result;
}

function error(err) { $('error').textContent = err?.message || '';connectionError=null;viewSelectionError=err?.filtered?err.message:null;rawSelectionError=err?.rawSelection?err.message:null; }
function projectorState() {
  const state=legacyMode&&legacyApi?legacyApi.state():{projection:$('projection').value,projectionReady:projectionWorker===null,
    availableProjections:[...$('projection').options].filter(o=>!o.disabled).map(o=>o.value),
    selectedTrackIds:[...selected],selectedCount:selected.size,pointCount:(projectedEmbedding||points).length};
  return {...state,embeddingPublications:embedding?.publications||[],coloring:pointColoring?{label:pointColoring.spec.label,kind:pointColoring.spec.kind,...(pointColoring.spec.palette?{palette:pointColoring.spec.palette}:{}),mappedCount:pointColoring.mappedCount}:null};
}
function setColoring(spec){
  pointColoring=createColoring(spec,[...new Set([...points.map(p=>p.key),...(embedding?.points||[]).map(p=>p.id)])]);
  updateTrajectoryMeasurement();legacyApi?.color(pointColoring?[...pointColoring.colors]:null);draw();
  const legend=$('color-legend');legend.replaceChildren();legend.hidden=!pointColoring;dockProjectionLegend(!!pointColoring);
  if(!pointColoring)return;
  const title=document.createElement('strong');title.textContent=spec.label+(spec.units?' ('+spec.units+')':'');legend.append(title);
  const items=document.createElement('div');items.className='color-entries';
  for(const entry of pointColoring.entries){const item=document.createElement('span'),swatch=document.createElement('i'),label=document.createElement('span');swatch.style.background=entry.color;label.className='color-label';label.textContent=entry.label;item.append(swatch,label);items.append(item);}
  if(spec.kind==='numeric'){const ramp=document.createElement('div');ramp.className='color-ramp';ramp.style.background='linear-gradient(90deg,'+pointColoring.ramp.join(',')+')';ramp.title=pointColoring.spec.palette+' · higher values are lighter';legend.append(ramp);}
  legend.append(items);legend.append(document.createElement('small'));updateLegendScope();
}
function projectionVisibleIds(){
  if(!legacyMode||!legacyApi){projectionViewRevision=null;projectionViewIds=null;return null;}
  const revision=legacyApi.viewRevision();
  if(revision!==projectionViewRevision){projectionViewRevision=revision;projectionViewIds=new Set(legacyApi.visibleTrackIds());}
  return projectionViewIds;
}
function filteredProjectionIds(){return legacyApi?.isFiltered?.()?projectionVisibleIds():null;}
function updateLegendScope(){
  updateSelectionScope();
  const note=$('color-legend').querySelector('small');if(!note||!pointColoring)return;
  // Raw paths and encoder vectors can cover different trajectories even when
  // no cohort is isolated. Count missing values only among the displayed IDs.
  const visible=projectionVisibleIds()||new Set(projectedEmbedding?projectedEmbedding.map(p=>p.id):points.map(p=>p.key));
  const missing=[...visible].filter(id=>!pointColoring.colors.has(id)).length;
  note.textContent=visible.size+' trajector'+(visible.size===1?'y':'ies')+' in this view'+(missing?' · '+missing+' without values (gray)':'')+' · color scale from '+pointColoring.mappedCount+' mapped trajector'+(pointColoring.mappedCount===1?'y':'ies')+' across the dataset';
  note.textContent+=$('projection').value==='raw'?' · Selected trajectories use thicker lines and larger points; colors retain supplied values.':' · Selected points are enlarged and labelled; colors retain supplied values.';
}
function updateSelectionScope(){
  if(rawSelectionError&&embedding&&[...selected].every(id=>embedding.points.some(p=>p.id===id))&&$('error').textContent===rawSelectionError)error(null);
  const scope=$('question-scope');scope.hidden=!files.length&&!selected.size;
  scope.textContent=selected.size?selected.size+' selected trajector'+(selected.size===1?'y accompanies':'ies accompany')+' your next question.':filteredProjectionIds()?'No trajectories selected. Your next question uses the full dataset. Choose Select all trajectories in this view to discuss this group.':'No trajectory selection is attached. Your next question uses the full dataset.';
}
function selectTracks(ids) {
  const knownIds=new Set([...points.map(t=>t.key),...(embedding?.points||[]).map(p=>p.id)]);
  if(!Array.isArray(ids)||ids.length>20000||ids.some(id=>typeof id!=='string'||!knownIds.has(id))) {
    const failure=new Error('Unknown track ID');failure.code='unknown_track';throw failure;
  }
  if(legacyMode&&legacyApi){
    const encoded=new Set(embedding.points.map(p=>p.id));
    if(ids.some(id=>!encoded.has(id))){const failure=new Error('This trajectory has no encoder vector in this projection. Switch to Raw trajectories to inspect it.');failure.code='unknown_track';throw failure;}
    const pending=pendingNativeSelection?.request===projectionGeneration?pendingNativeSelection:null,previous=pending?.ids;
    if(pending)pending.ids=[...ids];
    try{const result=legacyApi.select(ids);showTrajectory(ids);syncTrackChoice();if(active){saveWorkspaceDraft();renderSendHint();}return result;}catch{if(pending)pending.ids=previous;const failure=new Error('This trajectory is outside the current projection view. Restore all trajectories in View & parameters to inspect it.');failure.code='unknown_track';failure.filtered=true;throw failure;}
  }
  if(pendingNativeSelection?.request===projectionGeneration)pendingNativeSelection.ids=[...ids];
  selected=new Set(ids);draw();showTrajectory(ids);syncTrackChoice();if(active){saveWorkspaceDraft();renderSendHint();}return {selectedTrackIds:[...selected]};
}
function decode(data) { return new TextDecoder().decode(Uint8Array.from(atob(data), c => c.charCodeAt(0))); }
function encode(bytes) {
  let text = '';
  for (let i = 0; i < bytes.length; i += 8192) text += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(text);
}
function setFiles(value, preserveSelection=false) {
  const previousSelection=preserveSelection?[...selected]:[],previewWasOpen=!$('trajectory-card').hidden;
  files = value;if(files.length)missingRecoveryFiles=false;
  if(!active&&!sending)started=files.length>0;
  $('dataset-count').textContent=files.length?`${files.length} file${files.length===1?'':'s'}`:'Add trajectories';
  $('data-panel').open=!files.length;
  renderFileControls();
  preview();
  if(preserveSelection){const available=new Set(points.map(track=>track.key));selected=new Set(previousSelection.filter(id=>available.has(id)));draw();syncTrackChoice();}
  showJourney();if(!preserveSelection||previewWasOpen||!selected.size)showTrajectory([...selected]);
}
function renderFileControls() {
  const committed=!!(active&&latest?.files?.length);
  $('new').disabled=$('history').disabled=sending;
  for(const button of document.querySelectorAll('#conversation-list button,#draft-list button'))button.disabled=sending;
  $('send').disabled=sending||loadingWorkspace||unavailableWorkspace||missingAnalysisLink;
  $('start-new-analysis').disabled=sending;
  $('continue-analysis').disabled=$('retry-analysis').disabled=sending;
  $('recovery-note').hidden=!missingRecoveryFiles;
  $('files').disabled=$('sample').disabled=sending||loadingWorkspace||unavailableWorkspace||committed;
  $('dataset-lock').hidden=!committed;
  const storedDemo=latest?.turns?.find(t=>t.question.includes('\n\nDemo dataset context:\n'))?.question.split('\n\nDemo dataset context:\n')[1];
  $('demo-info').hidden=!(sampleFile&&files.includes(sampleFile))&&!storedDemo;
  if(storedDemo)$('demo-context').replaceChildren(renderAnswer(storedDemo));
  $('file-list').replaceChildren(...files.map(file => {
    const chip = document.createElement('span'); chip.className = 'file-chip';chip.append(document.createTextNode(file.name));
    if(!sending&&!unavailableWorkspace&&(!active||!latest?.files?.length)){
      const remove=document.createElement('button');remove.type='button';remove.textContent='×';remove.setAttribute('aria-label','Remove '+file.name);
      remove.onclick=()=>{if(sending||unavailableWorkspace||active&&latest?.files?.length)return;const remaining=files.filter(f=>f!==file),panelOpen=$('data-panel').open;if(active)pendingFiles=remaining;setFiles(remaining,true);$('data-panel').open=panelOpen;error(null);(files.length?$('data-panel').querySelector('summary'):$('files')).focus();};chip.append(remove);
    }
    return chip;
  }));
  renderSendHint();
}
function renderSendHint(){
  if(sending&&sendingWorkspaceKey!==workspaceKey())$('send-hint').textContent='Sending your earlier request in another workspace. Your current draft is preserved.';
  else if(missingAnalysisLink)$('send-hint').textContent='Start a new analysis here to send this question.';
  else if(unavailableWorkspace)$('send-hint').textContent='This unavailable analysis cannot accept follow-ups.';
  else if(!active)$('send-hint').textContent=files.length?'Unsent analysis. Your dataset will accompany your question.':'Include position units and time between frames, or try demo data.';
  else $('send-hint').textContent=['running','queued'].includes(latest?.status)?'Follow-ups will run after the current analysis.':'You can return to this conversation from this browser.';
  if(active&&!unavailableWorkspace&&!sending)$('send-hint').textContent+=viewRecoveryFailed?' View settings could not be saved for reload.':' View settings recover after reload in this tab; unsent questions do not.';
}
function dockProjectionLegend(open){
  const legend=$('color-legend'),dock=$('legend-dock'),plot=$('legacy-projector').parentElement;
  if(open){if(legend.parentElement!==dock)dock.append(legend);}
  else if(legend.parentElement!==plot)plot.insertBefore(legend,plot.querySelector('.stage-status'));
  dock.hidden=!open;
}
function resetEmbedding() {
  dockProjectionLegend(false);
  $('projection-notice').hidden=true;$('projection-notice').textContent='';
  pointColoring=null;colorVersion='null';$('color-legend').hidden=true;
  autoViewVersion=null;selectedCardId=null;previewSelectedGroup=false;trajectoryGroup=[];$('trajectory-card').hidden=true;
  projectionGeneration++;pendingNativeSelection=null;legacyApi?.cancel();legacyApi=null;legacyVersion=null;legacyMode=false;
  // A src change on an existing frame adds a nested browser-history entry.
  // Recreate its empty browsing context when changing datasets instead.
  const oldFrame=$('legacy-projector'),freshFrame=oldFrame.cloneNode(false);
  freshFrame.removeAttribute('src');freshFrame.hidden=true;oldFrame.replaceWith(freshFrame);$('plot').hidden=false;
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
async function setProjection(method,settings=null) {
  if(!['raw','pca','umap','tsne'].includes(method))throw new Error('Unknown projection.');
  // Returning to a raw view does not initialize the native projector. Keep its
  // saved cohort and controls available for the first switch back to it.
  if(!settings&&!legacyApi)settings=workspaceDrafts.get(workspaceKey())?.projectionSettings||null;
  const selectionPlan=method==='raw'||!embedding?null:planProjectionSelection([...selected],embedding.points.map(p=>p.id));
  if(selectionPlan?.unavailable.length){
    await setProjection('raw');
    const failure=new Error('Your selected trajectories do not all have encoder vectors. Their selection is preserved in raw view. Clear selection to explore the encoder projections.');
    failure.rawSelection=true;throw failure;
  }
  let selectionToSync=selectionPlan?.ids;
  updatePlotHelp(method);
  const request=++projectionGeneration;
  pendingNativeSelection={request,get ids(){return selectionToSync;},set ids(ids){selectionToSync=ids;}};
  legacyApi?.cancel();
  if(method==='raw'){
    pendingNativeSelection=null;
    dockProjectionLegend(!!pointColoring);
    $('projection-notice').hidden=true;
    for(const option of $('projection').options)option.disabled=option.value!=='raw'&&!embedding;
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
    if(!frame.getAttribute('src'))frame.src='./legacy-projector.html?v=15494e176060da07';
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
    legacyApi.setScreenshotHandler(async(canvas,snapshot)=>{
      const exportWorkspace=workspaceKey();
      const context=projectionFigureContext({snapshot,settings:legacyApi.viewSettings(),coloring:pointColoring,
        legendNote:$('color-legend').querySelector('small')?.textContent||'',filenames:files.map(f=>f.name)});
      try{
        if(!snapshot.projectionReady)throw new Error('Wait for the projection to finish before downloading its figure.');
        const blob=await renderProjectionFigure(canvas,context),url=URL.createObjectURL(blob),link=document.createElement('a');
        link.href=url;link.download='spt-'+snapshot.projection+'-projection.png';link.click();
        setTimeout(()=>URL.revokeObjectURL(url),60000);
        if(workspaceKey()===exportWorkspace){if(figureError&&$('error').textContent===figureError)error(null);figureError=null;}
      }catch(failure){if(workspaceKey()===exportWorkspace){figureError=failure.message;error(failure);}}
    });
    legacyApi.onChange=snapshot=>{
      if(!legacyMode)return;
      dockProjectionLegend(!!pointColoring);
      if(snapshot.controlsChanged){if(selectionToSync===null){saveWorkspaceDraft();renderSendHint();}return;}
      if(selectionToSync!==null&&snapshot.selectionChanged)selectionToSync=snapshot.selectedTrackIds;
      const ids=selectionToSync??snapshot.selectedTrackIds;
      const selectionChanged=ids.length!==selected.size||ids.some(id=>!selected.has(id));
      selected=new Set(ids);
      $('projection-notice').textContent=snapshot.projectionNotice||'';$('projection-notice').hidden=!snapshot.projectionNotice;
      for(const option of $('projection').options)option.disabled=!snapshot.availableProjections.includes(option.value);
      $('projection').value=snapshot.projection;
      updatePlotHelp(snapshot.projection);
      $('selection').textContent=selected.size?selected.size+' selected':'No trajectories selected';
      $('plot-meta').textContent=snapshot.pointCount+' trajector'+(snapshot.pointCount===1?'y':'ies');
      // Projection progress, camera state and warnings must respect a closed
      // preview. Explicit inspection and native point selections can reopen it.
      if(snapshot.selectionChanged||selectionChanged||!$('trajectory-card').hidden)showTrajectory(ids);
      updateTrackFinder(true);updateLegendScope();
      if(snapshot.inspectTrajectory&&!$('trajectory-card').hidden){trajectoryReturnFocus=$('legacy-projector');$('close-trajectory').focus();}
      if(snapshot.selectionChanged&&selectionToSync===null){saveWorkspaceDraft();renderSendHint();}
    };
    if(legacyVersion!==embeddingId){
      const version=embeddingId, usedApi=legacyApi;
      const rawById=new Map(points.map(p=>[p.key,p]));
      const payload={points:embedding.points.map(p=>({...p,xPosition:rawById.get(p.id)?.path.map(v=>v[0])||[],yPosition:rawById.get(p.id)?.path.map(v=>v[1])||[]}))};
      await usedApi.load(payload);
      if(request!==projectionGeneration||version!==embeddingId)return {cancelled:true};
      legacyVersion=version;
      usedApi.color(pointColoring?[...pointColoring.colors]:null);
    }
    if(request!==projectionGeneration)return {cancelled:true};
    if(settings)legacyApi.applyViewSettings(settings);
    const plan=planProjectionSelection(selectionToSync,embedding.points.map(p=>p.id),legacyApi.isFiltered()?legacyApi.visibleTrackIds():null);
    if(plan.restoreAll)legacyApi.restoreAll();
    legacyApi.select(plan.ids);
    const result=await legacyApi.project(method);
    if(request!==projectionGeneration||result.cancelled)return {cancelled:true};
    // Initial load starts in PCA; changing the native projection can clear its selection.
    legacyApi.select(selectionToSync??plan.ids);selectionToSync=null;pendingNativeSelection=null;
    legacyApi.color(pointColoring?[...pointColoring.colors]:null);
    $('projection').value=result.projection;updatePlotHelp(result.projection);$('plot-title').textContent='SPT projector';$('projection-settings').hidden=false;updateTrackFinder();updateLegendScope();
    if(active){saveWorkspaceDraft();renderSendHint();}
    return result;
  }catch(failure){
    if(request!==projectionGeneration)return {cancelled:true};
    pendingNativeSelection=null;
    console.warn('Native projection fallback:',failure);
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
  if(method==='raw') { $('plot-title').textContent='Trajectory preview'; preview(true); if(active){saveWorkspaceDraft();renderSendHint();}return {projection:'raw'}; }
  const small=embedding.points.length<3;
  if(small){
    method='pca';$('projection').value=method;updatePlotHelp(method);
    $('projection-notice').textContent='This dataset has '+embedding.points.length+' trajector'+(embedding.points.length===1?'y':'ies')+'. UMAP and t-SNE need at least 3 with these controls; showing PCA.'+(embedding.points.length===1?' Variance explained and comparisons between trajectories are unavailable.':'');
    $('projection-notice').hidden=false;
  }
  for(const option of $('projection').options)option.disabled=small&&['umap','tsne'].includes(option.value);
  $('plot-title').textContent=method==='tsne'?'t-SNE':method.toUpperCase();
  $('plot-meta').textContent='Computing in your browser…';
  const worker=new Worker(new URL('./projection-worker.js?v=15494e176060da07', import.meta.url),{type:'module'}); projectionWorker=worker;
  return new Promise((resolve,reject)=>{
    finishProjection=resolve;
    worker.onmessage=event=>{
      if(projectionWorker!==worker)return;
      const msg=event.data;
      if(msg.kind==='error'){worker.terminate();projectionWorker=null;finishProjection=null;error(new Error(msg.message));reject(new Error(msg.message));return;}
      if(msg.points){const first=!projectedEmbedding;projectedEmbedding=msg.points;if(first){updateTrackFinder();updateLegendScope();}draw();}
      if(msg.kind==='progress')$('plot-meta').textContent='Computing · iteration '+msg.iteration;
      if(msg.kind==='complete'){
        $('plot-meta').textContent=msg.points.length+' trajector'+(msg.points.length===1?'y':'ies')+' · encoder vectors';
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
    for(const f of chosen){
      const bytes=new Uint8Array(await f.arrayBuffer());let text;
      try{text=new TextDecoder('utf-8',{fatal:true}).decode(bytes);}catch{throw new Error(f.name+': this file is not valid UTF-8 text. Export or convert it to UTF-8 text before adding it. Binary files need a text export. Previously selected files are still available; none of this batch was added.');}
      if(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/.test(text))throw new Error(f.name+': this file contains binary or unsupported control characters. Export it as UTF-8 text before adding it. Previously selected files are still available; none of this batch was added.');
      added.push({name:f.name,bytes:f.size,data:encode(bytes)});
    }
    const combined=[...files,...added];if(active)pendingFiles=combined;setFiles(combined,true);$('data-panel').querySelector('summary').focus();
    error(null);
  } catch (err) { error(err); }
});
async function loadSample(stageQuestion=true) {
  if(sending&&stageQuestion)throw new Error('Wait for your request to finish sending.');
  if (files.length) throw new Error('Start a new analysis before replacing your dataset.');
  const questionBefore=$('question').value,workspaceBefore=workspaceKey();
  const sample = await api('/api/sample', undefined, null);
  if(workspaceKey()!==workspaceBefore||files.length)throw new Error('The dataset changed while the demo was loading. Try again.');
  sampleFile=sample.files[0];sampleContext=sample.question.split('## The question')[0];
  $('demo-context').replaceChildren(renderAnswer(sampleContext));
  if(active)pendingFiles=sample.files;setFiles(sample.files);
  if(stageQuestion&&!questionBefore.trim()&&!$('question').value.trim())$('question').value='Use demo data. What kind of motion do these particles exhibit?';
  error(null);
  return {...sample,questionStaged:stageQuestion};
}
$('sample').onclick = () => loadSample().catch(error);
$('new').onclick=()=>{if(sending)return;saveWorkspaceDraft();openUnsentWorkspace();sidebar(false);};
async function submitQuestion(question) {
  if (sending) throw new Error('A request is already being sent.');
  if(missingAnalysisLink)throw new Error('This browser cannot open the linked analysis. Choose Start a new analysis here to send this question separately.');
  if(unavailableWorkspace)throw new Error('Continue in a new analysis before sending another question.');
  if(loadingWorkspace)throw new Error('Wait for this analysis to load before sending a follow-up.');
  if (!question.trim()) throw new Error('Enter your question first.');

  const originKey=workspaceKey(),originJob=active,originCreationKey=creationKey;
  let submittedJob=originJob;
  sending = true;sendingWorkspaceKey=originKey; $('send').disabled = true;renderFileControls();
  try {
    let q = question.trim();
    const demoRequest=q.match(/\b(?:use|try|load|show|give)\b(?:(?!\b(?:no|without)\b)[^.!?]){0,50}\b(?:demo|sample)\b|^(?:demo|sample)(?: data)?[.!?]?$/i);
    const demoNegated=demoRequest&&/\b(?:don't|do not|not|never|without)\s+(?:\w+\s+){0,2}$/i.test(q.slice(0,demoRequest.index));
    if(!files.length&&demoRequest&&!demoNegated){
      await loadSample(false);
    }
    if(workspaceKey()!==originKey)throw new Error('The workspace changed before your request was sent. Return to its draft to retry.');
    if(sampleFile&&files.includes(sampleFile)&&(!originJob||pendingFiles.includes(sampleFile)))q+='\n\nDemo dataset context:\n'+sampleContext;
    started=true;showJourney();$('status').textContent='Sending your question…';
    const selection=selected.size?{ids:[...selected],totalCount:selected.size}:null;
    if (!originJob) {
      const created=await api('/api/jobs', {files:[...files], question:q, creationKey:originCreationKey,selection}, null);
      submittedJob={...created,title:question.trim().slice(0,70),sourceDraftKey:originKey};remember(submittedJob);
      if(workspaceKey()===originKey){active=submittedJob;sendingWorkspaceKey=active.id;writeWorkspaceAddress('replace');}
    } else {
      if (!pendingFollowup || pendingFollowup.job !== originJob.id || pendingFollowup.question !== q||JSON.stringify(pendingFollowup.selection)!==JSON.stringify(selection))
        pendingFollowup = {job:originJob.id, question:q, selection,requestId:crypto.randomUUID()};
      await api(`/api/jobs/${originJob.id}/messages`, {question:q, selection:pendingFollowup.selection,requestId:pendingFollowup.requestId,...(pendingFiles.length?{files:[...pendingFiles]}:{})},originJob);
      pendingFollowup=null;if(active?.id===originJob.id)pendingFiles=[];
    }
    const draft=workspaceDrafts.get(originKey);
    if(draft){
      const remaining={question:draft.question===question?'':draft.question,selection:draft.selection,projection:draft.projection,projectionSettings:draft.projectionSettings};
      workspaceDrafts.set(submittedJob.id,remaining);
    }
    if(!originJob)workspaceDrafts.delete(originKey);
    if(active?.id===submittedJob.id){
      if($('question').value===question)$('question').value='';
      error(null);await refresh();
    }
    renderDraftHistory();
    return {id:submittedJob.id,status:active?.id===submittedJob.id?latest?.status:'submitted'};
  } catch(err){
    if(workspaceKey()!==originKey&&active?.id!==submittedJob?.id)throw new Error('The request from the previous workspace could not be sent. Return to its draft to retry. '+err.message);
    if(!unavailableWorkspace)$('status').textContent='Submission interrupted · your draft is preserved';
    throw err;
  } finally { sending=false;sendingWorkspaceKey=null;if(!active)started=files.length>0;showJourney();renderFileControls(); }
}
$('composer').onsubmit = event => { event.preventDefault(); submitQuestion($('question').value).catch(error); };
$('cancel').onclick = async () => {
  try { await api(`/api/jobs/${active.id}/cancel`, {}); await refresh(); } catch(err) { error(err); }
};
function artifactOutcome(file,job=latest){
  const status=job?.turns?.[file.turn??0]?.status;
  if(status==='failed')return {label:'Request incomplete',detail:'This request did not finish. This file may be incomplete.'};
  if(status==='cancelled')return {label:'Request stopped',detail:'This request was stopped before completion. This file may be incomplete.'};
  return null;
}
function message(author, text, turn=Infinity) {
  const block = document.createElement('div'); block.className = 'message ' + (author === 'SPT AGENT' ? 'agent' : 'user');
  const label = document.createElement('span'); label.className = 'author'; label.textContent = author;
  block.append(label,author==='SPT AGENT'?renderAnswer(text,href=>resolveArtifact(href,turn),openArtifact):document.createTextNode(text)); return block;
}
function render(job) {
  if(recoveryDraftKey){workspaceDrafts.delete(recoveryDraftKey);resetUnavailable();renderDraftHistory();}
  latest = job;loadingWorkspace=false;
  if(active?.id===job.id){
    const question=typeof job.turns?.[0]?.question==='string'?questionForDisplay(job.turns[0]).text.trim():'';
    if(question)active.title=question.slice(0,70)+(question.length>70?'…':'');
    if(job.expiresAt)active.expiresAt=job.expiresAt;
    remember(active);
  }
  showJourney();
  const status = {queued:'Queued · waiting for the desktop', running:'Analyzing your dataset',
    completed:'Analysis complete',failed:'Analysis needs attention',cancelled:'Analysis stopped',
    cancelling:'Stopping analysis',interrupted:'Desktop connection interrupted'};
  $('status').textContent = status[job.status] || job.status;
  $('cancel').hidden = !['queued', 'running', 'cancelling'].includes(job.status);
  renderFileControls();
  renderSendHint();
  if (!files.length&&job.files.length) setFiles(job.files);
  const restoredDraft=draftToRestore;draftToRestore=null;
  if(restoredDraft)selected=new Set(restoredDraft.selection);
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
        const originalTracks=new Map(points.map(track=>[track.key,track]));
        const normalizedSources=new Set(candidate.points.filter(p=>p.xPosition?.length&&p.xPosition.length===p.yPosition?.length).map(p=>p.file));
        points=points.filter(p=>!p.fromEncoder&&!normalizedSources.has(p.file));
        const ids=new Set(points.map(p=>p.key));
        for(const p of candidate.points)if(!ids.has(p.id)&&p.xPosition?.length===p.yPosition?.length&&p.xPosition?.length){points.push(encoderPreviewTrack(p,originalTracks.get(p.id)));ids.add(p.id);}
        updateTrackFinder();
        for(const option of $('projection').options)option.disabled=false;
        $('embedding-info').textContent=candidate.points.length+' × '+candidate.points[0].vector.length+' encoder vectors';
        $('projection-progress').hidden=true;
        legacyVersion=null;autoViewVersion=candidate.version;embeddingId=candidate.version;
        setProjection(restoredDraft?.projection||'umap',restoredDraft?.projectionSettings).catch(error);
      }
      embeddingArtifactsVersion=vectorVersion;
    }catch(err){error(err);}
  }
  if(restoredDraft){
    if(!embedding)$('projection').value='raw';
    restoreDraftSelection(restoredDraft);
  }
  const nextColorVersion=JSON.stringify(job.pointColoring||null);
  if(nextColorVersion!==colorVersion){try{setColoring(job.pointColoring);colorVersion=nextColorVersion;}catch(err){error(err);}}
  const signature = JSON.stringify([job.updated, job.events.length, job.status]);
  if (signature === lastSignature) return;
  lastSignature = signature;
  const nearBottom = $('latest-message').hidden;
  const expanded=new Set([...$('messages').querySelectorAll('details[open][data-step]')].map(e=>e.dataset.step));
  const expandedSelections=new Set([...$('messages').querySelectorAll('details[open][data-selection]')].map(e=>e.dataset.selection));
  const progressEvents=job.events.filter(e=>e.kind==='summary'&&e.summary)
    .sort((a,b)=>a.turn-b.turn||a.order-b.order||a.id-b.id);
  const blocks=[];
  for(const [turn,t] of job.turns.entries()){
    const display=questionForDisplay(t),userBubble=message(t.requestId?.startsWith('operator-recovery-')?'RECOVERY':'YOU',display.text);
    if(display.selection){const badge=document.createElement('small'),s=display.selection;badge.className='selection-context';badge.textContent=s.legacy?'Trajectory selection attached':s.totalCount>s.ids.length?`First ${s.ids.length} of ${s.totalCount} selected trajectories attached`:`${s.ids.length} selected trajector${s.ids.length===1?'y':'ies'} attached`;userBubble.append(badge);}
    if(display.selection?.ids?.length){
      const details=document.createElement('details');details.className='selection-ids';details.dataset.selection=job.id+':'+turn;details.open=expandedSelections.has(details.dataset.selection);
      const summary=document.createElement('summary');summary.textContent='View attached trajectory IDs';
      const note=document.createElement('p');note.textContent='These IDs were attached at submission. Changing the current view does not change this question.';
      const ids=document.createElement('pre');ids.textContent=display.selection.ids.join('\n');
      details.append(summary,note,ids);userBubble.append(details);
    }
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
    if(['completed','failed','cancelled'].includes(t.status)){
      const bubble=message('SPT AGENT',t.answer||'',turn),note=document.createElement('p'),outcome=document.createElement('strong');
      note.className='response-status';note.dataset.status=t.status;
      outcome.textContent='Response '+(turn+1)+' · '+({completed:'Complete',failed:'Could not finish',cancelled:'Stopped'})[t.status];note.append(outcome);
      if(t.status!=='completed'){
        const detail=document.createElement('span');
        detail.textContent=t.status==='failed'?'This request did not finish. Any text or files from this response may be incomplete. You can ask a follow-up to continue.':'This request was stopped before completion. Any text or files from this response may be incomplete.';
        note.append(detail);
      }
      bubble.querySelector('.author').after(note);blocks.push(bubble);
    }
    else if(t.answer)blocks.push(message('SPT AGENT',t.answer,turn));
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
  const displayedArtifacts = listedResults(job.artifacts);
  for (const file of displayedArtifacts) {
    const version=artifactVersion(file,job.artifacts),outcome=artifactOutcome(file,job);
    const a=document.createElement('button');a.className='artifact';a.textContent=file.name+' · Response '+version.response+(version.superseded?' · Earlier version':'')+(outcome?' · '+outcome.label:'');a.onclick=()=>openArtifact(file);$('downloads').append(a);
  }
  $('result-count').textContent=displayedArtifacts.length?displayedArtifacts.length+' file'+(displayedArtifacts.length===1?'':'s'):'No outputs yet';
  $('bundle-controls').hidden=!displayedArtifacts.length;$('download-bundle').disabled=bundleBusy;
}
const bundleDownloadLink=document.createElement('a');bundleDownloadLink.hidden=true;document.body.append(bundleDownloadLink);
$('download-bundle').onclick=async()=>{
  const job=latest;if(!job||bundleBusy)return;bundleBusy=true;$('download-bundle').disabled=true;$('bundle-status').textContent='Preparing ZIP…';
  try{
    const method=$('projection').value,view={projection:method,parameters:method==='raw'?null:legacyApi?.viewSettings?.()||null,selectedTrackIds:[...selected],stochastic:method==='pca'?null:method==='umap'||method==='tsne'};
    const bundle=await buildAnalysisBundle(job,view);
    if(latest?.id!==job.id)return;
    if(bundleUrl)URL.revokeObjectURL(bundleUrl);bundleUrl=URL.createObjectURL(bundle.blob);
    bundleDownloadLink.href=bundleUrl;bundleDownloadLink.download=bundle.filename;bundleDownloadLink.click();
    $('bundle-status').textContent='ZIP download started. It contains '+bundle.manifest.files.filter(f=>f.kind==='input').length+' source dataset(s) and '+bundle.manifest.files.filter(f=>f.kind==='result').length+' result file(s).';
  }catch(failure){if(latest?.id===job.id)$('bundle-status').textContent=failure.message;}
  finally{bundleBusy=false;$('download-bundle').disabled=false;}
};
async function refresh() {
  clearTimeout(pollTimer);
  if (!active||unavailableWorkspace) return;
  const target = active;
  try {
    const job = await api('/api/jobs/' + target.id, undefined, target);
    if (active?.id === target.id&&!unavailableWorkspace) {
      if(connectionError&&$('error').textContent===connectionError)error(null);
      render(job);
      projectorBridge.tick(job,target).catch(()=>{}); // Cached result retries on next poll.
    }
  } catch(err) {
    if(active?.id===target.id&&!unavailableWorkspace){
      const message='Refresh interrupted. Retrying; your draft is preserved. '+err.message;
      error(new Error(message));connectionError=message;
    }
  }
  finally { if (active&&!unavailableWorkspace) pollTimer = setTimeout(refresh, cloudPollDelay(latest,document.hidden)); }
}

// Spatial preview is explicitly raw trajectories, never a fabricated embedding.
function preview(keepSelection=false) {
  const originalPoints=points;points=[];if(!keepSelection)selected.clear();
  const warnings=[];
  for (const file of files) {
    const normalizedPreview=keepSelection&&originalPoints.some(point=>point.fromEncoder&&point.file===file.name);
    if (!/\.(csv|tsv)$/i.test(file.name)) {if(!normalizedPreview)warnings.push(file.name+': automatic browser preview supports CSV/TSV files. This file remains in the dataset; describe its format, columns and units in your question.');continue;}
    // Large datasets are preprocessed by the agent; avoid splitting millions of rows on the UI thread.
    if ((file.bytes??file.data.length*3/4)>16*1024*1024){warnings.push(file.name+': automatic preview of the original file is deferred above 16 MiB. The complete file remains in the dataset for analysis.');continue;}
    try{
      const parsed=parseTrajectoryTable(decode(file.data),file.name);points.push(...parsed.tracks);
      if(parsed.empty)warnings.push(file.name+': no header or observation rows were found. Check that the export contains data.');
      if(parsed.headerOnly)warnings.push(file.name+': trajectory headers were found, but there are no observation rows. Check that the export contains data.');
      if(parsed.delimiterMismatch)warnings.push(file.name+': preview detected '+parsed.delimiter+'-separated data despite its file extension. The trajectory columns were recognized.');
      if(parsed.commaNumberRows)warnings.push(file.name+': preview found '+parsed.commaNumberRows+' row'+(parsed.commaNumberRows===1?'':'s')+' with comma-formatted coordinates or timestamps. These values were not interpreted: use decimal points and no thousands separators for numeric preview fields. Describe the original number format in your question.');
      if(parsed.missingColumns?.length&&!normalizedPreview)warnings.push(file.name+': not shown in this preview because its headers do not include '+parsed.missingColumns.join(', ')+'. Describe its columns and units in your question so the agent can normalize it. Do not rename pixel coordinates to x_um/y_um without converting to micrometers.');
      if(parsed.tracks.length&&!parsed.hasTimestamps)warnings.push(file.name+': spatial preview only; no t_s timestamp column was found. Describe the time between frames and any gaps before asking for motion measurements.');
      if(parsed.tracks.some(track=>track.hasZColumn))warnings.push(file.name+': z_um is present. This preview and its animation show only the XY projection; inspect an observation to read its z value. Ask for XYZ measurements to include z in the analysis.');
      if(parsed.malformedRows)warnings.push(file.name+': preview skipped '+parsed.malformedRows+' row'+(parsed.malformedRows===1?'':'s')+' whose field count does not match the '+parsed.columnCount+' column headers. Check for extra or missing separators, and quote text that contains a separator.');
      const invalidRows=parsed.skippedRows-(parsed.malformedRows||0);
      if(invalidRows)warnings.push(file.name+': preview skipped '+invalidRows+' row'+(invalidRows===1?'':'s')+' with missing or invalid IDs, coordinates or timestamps.');
      if(parsed.duplicateTimes)warnings.push(file.name+': preview found '+parsed.duplicateTimes+' repeated track/timestamp pair'+(parsed.duplicateTimes===1?'':'s')+' across '+parsed.duplicateTimeTracks+' trajector'+(parsed.duplicateTimeTracks===1?'y':'ies')+'. All valid observations were retained. Check for duplicate frames or IDs reused across cells or movies; qualify reused IDs before interpreting combined paths.');
      if(parsed.reorderedTracks)warnings.push(file.name+': preview ordered observations by timestamp in '+parsed.reorderedTracks+' trajector'+(parsed.reorderedTracks===1?'y':'ies')+' whose source rows were out of order.');
    }catch(failure){warnings.push(file.name+': spatial preview unavailable. '+failure.message);}
  }
  // Switching to raw view must retain agent-normalized paths from noncanonical files.
  points=rawPreviewTracks(points,originalPoints,keepSelection);
  $('preview-warning').textContent=warnings.join(' ')+(warnings.length?' The original uploaded files are unchanged.':'');$('preview-warning').hidden=!warnings.length;
  $('plot-empty').hidden=points.length>0;
  $('plot-empty').textContent=files.length?'No spatial trajectories are available for an automatic browser preview. The agent can still inspect and preprocess these files.':'Upload trajectories to inspect their paths. You can ask a question without selecting points.';
  $('plot-meta').textContent=points.length?`${points.length} trajector${points.length===1?'y':'ies'} · x/y in µm`:'No spatial preview';
  updateTrackFinder();updateLegendScope();draw();
}
let finderTracks=[],finderScopeIds;
function updateTrackFinder(scopeOnly=false){
  const visible=projectionVisibleIds();
  const filtered=filteredProjectionIds();
  if(!filtered&&viewSelectionError&&$('error').textContent===viewSelectionError)error(null);
  if(scopeOnly&&visible===finderScopeIds){syncTrackChoice();return;}
  finderScopeIds=visible;
  const byId=new Map(points.map(track=>[track.key,track]));
  for(const point of embedding?.points||[])if(!byId.has(point.id))byId.set(point.id,{key:point.id,id:point.trackId,file:point.file});
  finderTracks=[...byId.values()].sort((a,b)=>a.key.localeCompare(b.key));
  const shown=visible||new Set(projectedEmbedding?projectedEmbedding.map(track=>track.id):points.map(track=>track.key));
  finderTracks=finderTracks.filter(track=>shown.has(track.key));
  $('restore-tracks').hidden=$('view-scope').hidden=!filtered;
  $('view-scope').textContent=filtered?filtered.size+' of '+byId.size+' trajectories shown. Hidden trajectories remain in your dataset.':'';
  $('select-visible').hidden=$('visible-selection-help').hidden=!finderTracks.length;
  $('select-visible').textContent='Select all '+finderTracks.length+' trajectories in this view';
  $('select-visible').disabled=finderTracks.length>20000;
  $('visible-selection-help').textContent=finderTracks.length>20000?'Up to 20,000 trajectories can accompany a question. Use a smaller group to select them together.':'Includes all trajectories in this view, regardless of zoom or picker search. Replaces your selection.';
  $('track-finder').hidden=!finderTracks.length;
  if(!finderTracks.length)$('track-search').value='';
  renderTrackChoices();
}
function renderTrackChoices(){
  const matches=matchingTracks(finderTracks,$('track-search').value);
  const options=[Object.assign(document.createElement('option'),{value:'',textContent:'Choose a trajectory'})];
  for(const track of matches.tracks)options.push(Object.assign(document.createElement('option'),{value:track.key,textContent:track.key}));
  $('track-choice').replaceChildren(...options);syncTrackChoice();
  $('track-matches').textContent=matches.total>matches.tracks.length?'Showing the first '+matches.tracks.length+' of '+matches.total+' matches. Refine the filename or track ID.':matches.total+' matching trajector'+(matches.total===1?'y':'ies')+'. Inspecting one replaces the current selection.';
}
function syncTrackChoice(){ $('track-choice').value=selected.size===1?[...selected][0]:'';$('track-inspect').disabled=!$('track-choice').value;$('preview-selected').hidden=$('preview-selected-help').hidden=selected.size<2; }
$('track-search').oninput=renderTrackChoices;
$('track-choice').onchange=()=>{$('track-inspect').disabled=!$('track-choice').value;};
$('restore-tracks').onclick=()=>{try{legacyApi.restoreAll();updateTrackFinder();updateLegendScope();error(null);$('track-search').focus();}catch(failure){error(failure);}};
$('select-visible').onclick=()=>{try{selectTracks(finderTracks.map(track=>track.key));$('view-panel').open=false;$('question').focus();}catch(failure){error(failure);}};
$('preview-selected').onclick=()=>{showTrajectory([...selected],true,$('track-choice').value);$('view-panel').open=false;$('trajectory-next').focus();};
$('track-inspect').onclick=()=>{try{selectTracks([$('track-choice').value]);$('view-panel').open=false;$('close-trajectory').focus();}catch(failure){syncTrackChoice();error(failure);}};
$('view-panel').onkeydown=event=>{if(event.key==='Escape'){$('view-panel').open=false;$('view-panel').querySelector('summary').focus();}};
const canvas=$('plot');let projected=[];
function updatePlotHelp(method=$('projection').value){
  $('plot-help').textContent=method==='raw'?'Click a trajectory to inspect it, or find its filename and ID in View & parameters. Drag a lasso to select several. Selected tracks accompany your next question.':'Each point represents one trajectory’s encoder vector. Projection distances have no spatial units and clusters alone do not establish a motion model. Click a point to inspect it, or find its filename and ID in View & parameters.';
  if(method!=='raw')$('plot-help').textContent+=' In 3D, Rotate view moves only the camera; Pause rotation keeps your selection.';
  if(method!=='raw'&&legacyMode)$('plot-help').textContent+=' Box selection includes overlapping points whose centers lie inside the box.';
  if(method==='umap'||method==='tsne')$('plot-help').textContent+=' This projection is stochastic; rerunning can change the layout even with the same parameters.';
  if(method==='pca'&&legacyMode)$('plot-help').textContent+=' This PCA uses uncentered SVD; component percentages describe squared vector magnitude, including the mean. See Projection settings for details.';
}
function draw(){
  updateSelectionScope();
  updatePlotHelp();
  const box=canvas.getBoundingClientRect();const scale=devicePixelRatio||1;
  canvas.width=box.width*scale;canvas.height=box.height*scale;
  const c=canvas.getContext('2d');c.scale(scale,scale);c.clearRect(0,0,box.width,box.height);
  projected=[];
  $('selection').textContent=selected.size?`${selected.size} selected trajector${selected.size===1?'y':'ies'}`:'No trajectories selected';
  c.strokeStyle='#e6eaf0';c.lineWidth=.5;
  for(let i=1;i<8;i++){c.beginPath();c.moveTo(i*box.width/8,0);c.lineTo(i*box.width/8,box.height);c.stroke();c.beginPath();c.moveTo(0,i*box.height/8);c.lineTo(box.width,i*box.height/8);c.stroke();}
  const visible=projectedEmbedding ? projectedEmbedding.map(p=>({id:p.id,key:p.id,path:[[p.x,p.y]]})) : points;
  if(!visible.length)return;
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
  for(const t of visible)for(const [x,y]of t.path){minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);}
  const s=Math.min((box.width-48)/(maxX-minX||1),(box.height-48)/(maxY-minY||1));
  projected=visible.map((t,i)=>{
    const path=t.path.map(([x,y])=>[24+(x-minX)*s,box.height-24-(y-minY)*s]);
    c.strokeStyle=pointColoring?(pointColoring.colors.get(t.key)||'#b7bec8'):selected.size?(selected.has(t.key)?'#d6584f':'#9ca9be'):['#547cce','#388c94','#9671b2','#b18745'][i%4];
    c.globalAlpha=!pointColoring&&selected.size&&!selected.has(t.key)?.25:.8;c.lineWidth=selected.has(t.key)?(pointColoring?2.6:1.6):.9;c.beginPath();
    path.forEach(([x,y],j)=>j?c.lineTo(x,y):c.moveTo(x,y));c.stroke();
    if(projectedEmbedding||t.path.length>0&&t.path.every(([x,y])=>x===t.path[0][0]&&y===t.path[0][1])){c.fillStyle=c.strokeStyle;c.beginPath();c.arc(path[0][0],path[0][1],selected.has(t.key)?(pointColoring?5:4):3,0,Math.PI*2);c.fill();}
    return {...t,path};
  });c.globalAlpha=1;
  if(polygon.length){c.beginPath();polygon.forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.strokeStyle='#4779dc';c.lineWidth=1.5;c.stroke();}
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
  drawing=false;polygon=[];draw();showTrajectory([...selected]);syncTrackChoice();
};
canvas.onpointercancel=()=>{drawing=false;polygon=[];draw();};
$('clear-selection').onclick=()=>{selectTracks([]);showTrajectory([]);};
$('projection-settings').onclick=()=>{$('view-panel').open=false;legacyApi?.openParameters();};
new ResizeObserver(draw).observe(canvas);
let hiddenConversationScroll=0,hiddenConversationWorkspace=null;
function panels(data,chat){
  const wasChatHidden=$('conversation').hidden;
  if(!chat&&!wasChatHidden){hiddenConversationScroll=$('messages').scrollTop;hiddenConversationWorkspace=workspaceKey();}
  const restoreScroll=chat&&wasChatHidden&&hiddenConversationWorkspace===workspaceKey()?hiddenConversationScroll:null;
  const restoreWorkspace=workspaceKey();
  $('explorer').hidden=!data;$('conversation').hidden=!chat;$('show-data').hidden=data;$('show-chat').hidden=chat;main.classList.toggle('data-hidden',!data);main.classList.toggle('chat-hidden',!chat);
  if(restoreScroll!==null)$('messages').scrollTop=restoreScroll;
  requestAnimationFrame(()=>{draw();if(restoreScroll!==null&&!$('conversation').hidden&&workspaceKey()===restoreWorkspace)$('messages').scrollTop=restoreScroll;});
}
$('hide-data').onclick=()=>{panels(false,true);$('show-data').focus();};
$('show-data').onclick=()=>{panels(true,true);$('hide-data').focus();};
$('hide-chat').onclick=()=>{panels(true,false);$('show-chat').focus();};
$('show-chat').onclick=()=>{panels(true,true);$('hide-chat').focus();};

if(document.modelContext?.registerTool){
  const lifecycle=new AbortController();
  const register=t=>Promise.resolve(document.modelContext.registerTool(t,{signal:lifecycle.signal})).catch(()=>{});
  register({name:'read_spt_workspace',description:'Read current dataset, job status, and selected track IDs.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true},execute:()=>({files:files.map(f=>f.name),job:active?.id,status:latest?.status,selectedTrackIds:[...selected]})});
  register({name:'stage_spt_sample',description:'Load the sample dataset and stage its question without starting analysis.',inputSchema:{type:'object',properties:{},additionalProperties:false},execute:loadSample});
  register({name:'set_spt_projection',description:'Compute and show PCA, UMAP, or t-SNE from the available encoder vectors, or show raw trajectories.',inputSchema:{type:'object',properties:{method:{type:'string',enum:['raw','pca','umap','tsne']}},required:['method'],additionalProperties:false},execute:input=>setProjection(input?.method)});
  register({name:'select_spt_tracks',description:'Select file-qualified track IDs (filename:track_id) in the visible trajectory preview.',inputSchema:{type:'object',properties:{ids:{type:'array',items:{type:'string'}}},required:['ids'],additionalProperties:false},execute:input=>selectTracks(input?.ids)});
  window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
}
function resolveArtifact(href,turn=Infinity){return resolveArtifactLink(href,latest?.artifacts||[],turn);}
function openArtifact(file){
  artifactUrls.forEach(url=>URL.revokeObjectURL(url));artifactUrls=[];
  const bytes=Uint8Array.from(atob(file.data),c=>c.charCodeAt(0)),ext=file.name.split('.').at(-1).toLowerCase();
  const type=({png:'image/png',jpg:'image/jpeg',svg:'image/svg+xml',pdf:'application/pdf',csv:'text/csv'})[ext]||'text/plain';
  const url=URL.createObjectURL(new Blob([bytes],{type}));artifactUrls.push(url);
  $('artifact-title').textContent=file.name;$('artifact-download').href=url;$('artifact-download').download=file.name;
  const version=artifactVersion(file,latest?.artifacts||[]),notice=$('artifact-version');
  notice.replaceChildren(document.createTextNode('Response '+version.response+(version.superseded?' · Earlier version. A newer version of this file is available.':' · Latest version.')));
  const outcome=artifactOutcome(file);if(outcome){const detail=document.createElement('span');detail.className='artifact-outcome';detail.textContent=outcome.detail;notice.append(detail);}
  if(version.superseded){const button=document.createElement('button');button.className='quiet';button.textContent='Open latest version';button.onclick=()=>{openArtifact(version.latest);$('artifact-download').focus();};notice.append(button);}
  const preview=$('artifact-preview');preview.replaceChildren();
  $('figure-view-controls').hidden=true;
  preview.removeAttribute('tabindex');preview.removeAttribute('role');preview.removeAttribute('aria-label');preview.removeAttribute('aria-describedby');
  if(['png','jpg','svg'].includes(ext)){
    const img=document.createElement('img'),fit=$('figure-fit'),original=$('figure-original'),help=$('figure-view-help');
    img.alt=file.name;preview.append(img);$('figure-view-controls').hidden=false;
    fit.disabled=original.disabled=true;fit.setAttribute('aria-pressed','true');original.setAttribute('aria-pressed','false');help.textContent='Loading figure…';
    preview.tabIndex=0;preview.setAttribute('role','region');preview.setAttribute('aria-label','Figure preview: '+file.name);preview.setAttribute('aria-describedby','figure-view-help');
    const size=(actual,event)=>{
      img.classList.toggle('original-size',actual);fit.setAttribute('aria-pressed',String(!actual));original.setAttribute('aria-pressed',String(actual));
      help.textContent=actual?(ext==='svg'?'Original SVG viewport: ':'Original image size: ')+img.naturalWidth+' × '+img.naturalHeight+(ext==='svg'?' CSS pixels.':' pixels.')+' Scroll to inspect details; choose Fit figure to see the whole figure.':'Fitted to the preview. Choose Original size to inspect axis labels and other details.';
      preview.scrollTop=preview.scrollLeft=0;
      if(event?.detail===0)preview.focus();
    };
    fit.onclick=event=>size(false,event);original.onclick=event=>size(true,event);
    img.onload=()=>{if(!img.isConnected)return;fit.disabled=original.disabled=false;size(false);};
    img.onerror=()=>{if(img.isConnected)help.textContent='This figure could not be displayed. Use Download to open the original file.';};
    img.src=url;
  }
  else if(ext==='pdf'){const p=document.createElement('p');p.textContent='Download this PDF to open it in your document viewer.';preview.append(p);}
  else {const text=new TextDecoder().decode(bytes);if(ext==='md')preview.append(renderAnswer(text,href=>resolveArtifact(href,file.turn),openArtifact));else{const pre=document.createElement('pre');pre.textContent=text.slice(0,200000)+(text.length>200000?'\n… Download the full file to read more.':'');preview.append(pre);if(text.length>200000)notice.append(Object.assign(document.createElement('span'),{className:'artifact-preview-limit',textContent:'This text preview is shortened. Use Download for the complete file.'}));}}
  if(!$('artifact-dialog').open)$('artifact-dialog').showModal();
  preview.scrollTop=0;preview.scrollLeft=0;$('artifact-dialog').scrollTop=0;
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
let trajectoryFrame=0,trajectoryRunning=true,trajectoryTimer=null,trajectoryPath=[],trajectoryTimes=[],trajectoryZValues=null,trajectoryReturnFocus=null;
let trajectoryGroup=[],previewSelectedGroup=false,trajectoryChoicesKey=null;
function renderTrajectoryChoices(id=selectedCardId,force=false){
  const key=JSON.stringify([trajectoryGroup,$('trajectory-search').value]);
  if(!force&&key===trajectoryChoicesKey)return;
  trajectoryChoicesKey=key;
  const matches=matchingTracks(trajectoryGroup.map(key=>({key})),$('trajectory-search').value);
  const options=[Object.assign(document.createElement('option'),{value:'',textContent:'Choose a selected trajectory'})];
  for(const track of matches.tracks)options.push(Object.assign(document.createElement('option'),{value:track.key,textContent:track.key}));
  $('trajectory-choice').replaceChildren(...options);
  $('trajectory-choice').value=matches.tracks.some(track=>track.key===id)?id:'';
  $('trajectory-jump').disabled=!$('trajectory-choice').value;
  $('trajectory-matches').textContent=matches.total>matches.tracks.length?'Showing the first '+matches.tracks.length+' of '+matches.total+' selected matches. Refine the filename or track ID.':matches.total+' matching selected trajector'+(matches.total===1?'y.':'ies.');
}
$('trajectory-search').oninput=()=>renderTrajectoryChoices(selectedCardId,true);
$('trajectory-choice').onchange=()=>{$('trajectory-jump').disabled=!$('trajectory-choice').value;};
$('trajectory-jump').onclick=()=>{const id=$('trajectory-choice').value;if(selected.has(id))showTrajectory([...selected],true,id);};
function updateTrajectoryMeasurement(){
  const line=$('trajectory-measurement'),spec=pointColoring?.spec;
  line.hidden=!spec||!selectedCardId;
  if(line.hidden){line.textContent='';return;}
  const present=Object.hasOwn(spec.values,selectedCardId);
  line.textContent=spec.label+': '+(present?String(spec.values[selectedCardId])+(spec.units?' '+spec.units:''):'No value supplied for this trajectory.');
}
function showTrajectory(ids,openGroup=false,previewId=null){
  ids=[...new Set(ids)];
  if(openGroup&&ids.length>1)previewSelectedGroup=true;
  if(!ids.length||(ids.length>1&&!previewSelectedGroup)){
    const hadFocus=$('trajectory-card').contains(document.activeElement);
    $('trajectory-card').hidden=true;selectedCardId=null;trajectoryGroup=[];previewSelectedGroup=false;$('trajectory-search').value='';trajectoryChoicesKey=null;clearTimeout(trajectoryTimer);
    if(hadFocus)$('view-panel').querySelector('summary').focus();return;
  }
  if(ids.length===1)previewSelectedGroup=false;
  trajectoryGroup=[...ids];
  const id=previewId&&ids.includes(previewId)?previewId:ids.includes(selectedCardId)?selectedCardId:ids[0];
  $('trajectory-group').hidden=ids.length<2;
  if(ids.length<2){$('trajectory-search').value='';trajectoryChoicesKey=null;}
  else renderTrajectoryChoices(id,selectedCardId!==id);
  if(ids.length<2&&$('trajectory-group').contains(document.activeElement))$('close-trajectory').focus();
  const position='Previewing '+(ids.indexOf(id)+1)+' of '+ids.length+' selected trajectories. All '+ids.length+' remain attached to your next question.';
  if($('trajectory-position').textContent!==position)$('trajectory-position').textContent=position;
  const track=points.find(t=>t.key===id),vector=embedding?.points.find(p=>p.id===id);
  if(!track&&!vector){$('trajectory-card').hidden=true;selectedCardId=null;clearTimeout(trajectoryTimer);return;}
  if(selectedCardId===id&&!$('trajectory-card').hidden)return;
  const focused=document.activeElement;
  if(!$('trajectory-card').contains(focused))trajectoryReturnFocus=focused===document.body?null:focused;
  selectedCardId=id;trajectoryPath=track?.path||[];trajectoryTimes=track?.times||[];trajectoryZValues=track?.hasZColumn?track.zValues||[]:null;trajectoryFrame=0;trajectoryRunning=true;
  updateTrajectoryMeasurement();$('trajectory-title').textContent=id;$('trajectory-card').hidden=false;$('trajectory-card').scrollTop=0;$('trajectory-play').textContent='Pause';
  const available=!!trajectoryPath.length;
  $('trajectory-note').textContent='Each trajectory is fitted independently to this preview. Playback uses uniform observation steps. '+(trajectoryTimes.length?'Read the displayed timestamps for acquisition timing, including gaps and repeats.':'No timestamps are available for this trajectory; playback shows observation order only.');
  $('trajectory-source').textContent='File: '+(track?.file||vector.file)+' · Track ID: '+(track?.id??vector.trackId)+(available?' · '+trajectoryPath.length+(trajectoryPath.length===1?' observation':' observations')+(track?.hasZColumn?' · x/y/z in µm · Animation shows the XY projection only':' · x/y in µm')+(trajectoryTimes.length?' · t = '+trajectoryTimes[0]+' to '+trajectoryTimes.at(-1)+' s':''):' · Raw spatial observations are unavailable for this encoder vector.');
  $('trajectory-animation').hidden=$('trajectory-play').hidden=$('trajectory-frame').hidden=$('trajectory-note').hidden=$('trajectory-observation-controls').hidden=!available;
  for(const input of [$('trajectory-timeline'),$('trajectory-observation')]){input.max=String(trajectoryPath.length||1);input.value='1';input.disabled=trajectoryPath.length<2;}
  $('trajectory-observation-go').disabled=trajectoryPath.length<2;
  clearTimeout(trajectoryTimer);if(available)animateTrajectory();syncTrackChoice();
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
  $('trajectory-frame').textContent=`Observation ${trajectoryFrame+1} / ${coords.length}`+(trajectoryTimes.length?` · t = ${trajectoryTimes[trajectoryFrame]} s`:'');
  const observation=trajectoryFrame+1;
  $('trajectory-timeline').value=String(observation);
  $('trajectory-timeline').setAttribute('aria-valuetext','Observation '+observation+' of '+coords.length+(trajectoryTimes.length?', t = '+trajectoryTimes[trajectoryFrame]+' s':''));
  if(document.activeElement!==$('trajectory-observation'))$('trajectory-observation').value=String(observation);
  const [rawX,rawY]=trajectoryPath[trajectoryFrame];
  $('trajectory-coordinates').textContent='x = '+rawX+' µm · y = '+rawY+' µm'+(trajectoryZValues===null?'':Number.isFinite(trajectoryZValues[trajectoryFrame])?' · z = '+trajectoryZValues[trajectoryFrame]+' µm':' · z unavailable (missing or invalid z_um)');
  if(trajectoryRunning)trajectoryTimer=setTimeout(()=>{trajectoryFrame=(trajectoryFrame+1)%coords.length;animateTrajectory();},Math.max(16,4000/coords.length));
}
function closeTrajectory(){
  $('trajectory-card').hidden=true;selectedCardId=null;trajectoryGroup=[];previewSelectedGroup=false;$('trajectory-search').value='';trajectoryChoicesKey=null;clearTimeout(trajectoryTimer);
  const target=trajectoryReturnFocus;trajectoryReturnFocus=null;
  if(target?.isConnected&&!target.disabled&&!target.closest('[hidden]')){
    if(target===$('legacy-projector')&&legacyMode&&legacyApi?.focusSelectionDetails())return;
    if($('view-panel').contains(target)&&target!==$('view-panel').querySelector('summary'))$('view-panel').open=true;
    if(target.getClientRects().length){target.focus();return;}
  }
  $('view-panel').querySelector('summary').focus();
}
function stepTrajectory(direction){
  if(trajectoryGroup.length<2)return;
  const index=(trajectoryGroup.indexOf(selectedCardId)+direction+trajectoryGroup.length)%trajectoryGroup.length;
  showTrajectory([...selected],true,trajectoryGroup[index]);
}
$('trajectory-previous').onclick=()=>stepTrajectory(-1);
$('trajectory-next').onclick=()=>stepTrajectory(1);
$('close-trajectory').onclick=closeTrajectory;
$('trajectory-card').onkeydown=event=>{if(event.key==='Escape'){event.preventDefault();event.stopPropagation();closeTrajectory();}};
function pauseTrajectory(){trajectoryRunning=false;clearTimeout(trajectoryTimer);$('trajectory-play').textContent='Play';}
function seekTrajectory(observation){
  if(!Number.isInteger(observation)||observation<1||observation>trajectoryPath.length)return;
  pauseTrajectory();trajectoryFrame=observation-1;animateTrajectory();
}
$('trajectory-timeline').onfocus=$('trajectory-observation').onfocus=pauseTrajectory;
$('trajectory-timeline').oninput=()=>seekTrajectory(Number($('trajectory-timeline').value));
$('trajectory-observation').oninput=pauseTrajectory;
function jumpObservation(){const input=$('trajectory-observation');if(input.reportValidity()){seekTrajectory(Number(input.value));input.value=String(trajectoryFrame+1);}}
$('trajectory-observation-go').onclick=jumpObservation;
$('trajectory-observation').onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();jumpObservation();}};
$('trajectory-play').onclick=()=>{trajectoryRunning=!trajectoryRunning;$('trajectory-play').textContent=trajectoryRunning?'Pause':'Play';clearTimeout(trajectoryTimer);if(trajectoryRunning)animateTrajectory();};
if(!missingAnalysisLink)writeWorkspaceAddress('replace');
else {history.replaceState({sptWorkspace:workspaceKey()},'',location.href);lastBrowserAddress=browserAddress();}
renderHistory();showJourney();
if(active){draftToRestore=viewRecovery.load(active.id);if(draftToRestore)workspaceDrafts.set(active.id,draftToRestore);loadingWorkspace=true;$('status').textContent='Loading analysis…';renderFileControls();refresh();}else {
  draw();
  renderFileControls();
}

document.addEventListener('visibilitychange',()=>{if(!document.hidden&&active)refresh();});
