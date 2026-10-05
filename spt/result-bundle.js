import {artifactVersion} from './artifact-links.js?v=9e40346669975800';

// Stored ZIP records, UTF-8 names and CRC-32 per PKWARE APPNOTE 6.3.10.
// https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
const encoder=new TextEncoder(),limit=512*1024*1024;
const crcTable=Uint32Array.from({length:256},(_,n)=>{for(let k=0;k<8;k++)n=n&1?0xedb88320^(n>>>1):n>>>1;return n>>>0;});
function crc32(bytes){let crc=0xffffffff;for(const byte of bytes)crc=crcTable[(crc^byte)&255]^(crc>>>8);return (crc^0xffffffff)>>>0;}
const decode=data=>Uint8Array.from(atob(data),c=>c.charCodeAt(0));
function safeName(name){
  const clean=String(name).normalize('NFC').replace(/[\x00-\x1f\x7f/\\:<>"|?*]/g,'_').replace(/^[. ]+|[. ]+$/g,'')||'file';
  return /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(clean)?'_'+clean:clean;
}
function allocatePath(directory,name,used){
  const clean=safeName(name),dot=clean.lastIndexOf('.'),stem=dot>0?clean.slice(0,dot):clean,ext=dot>0?clean.slice(dot):'';
  let path=directory+'/'+clean,n=1;
  while(used.has(path.toLowerCase()))path=directory+'/'+stem+'__'+(++n)+ext;
  used.add(path.toLowerCase());return path;
}
function zip(entries){
  if(entries.length>=65535)throw new Error('Too many files for one ZIP. Download individual results.');
  const parts=[],central=[];let offset=0;
  for(const {path,bytes} of entries){
    const name=encoder.encode(path),crc=crc32(bytes);
    if(name.length>65535)throw new Error('A filename is too long for a ZIP. Download individual results.');
    const local=new Uint8Array(30+name.length),lv=new DataView(local.buffer);
    lv.setUint32(0,0x04034b50,true);lv.setUint16(4,20,true);lv.setUint16(6,0x800,true);lv.setUint16(12,33,true);
    lv.setUint32(14,crc,true);lv.setUint32(18,bytes.length,true);lv.setUint32(22,bytes.length,true);lv.setUint16(26,name.length,true);local.set(name,30);
    const record=new Uint8Array(46+name.length),cv=new DataView(record.buffer);
    cv.setUint32(0,0x02014b50,true);cv.setUint16(4,20,true);cv.setUint16(6,20,true);cv.setUint16(8,0x800,true);cv.setUint16(14,33,true);
    cv.setUint32(16,crc,true);cv.setUint32(20,bytes.length,true);cv.setUint32(24,bytes.length,true);cv.setUint16(28,name.length,true);cv.setUint32(42,offset,true);record.set(name,46);
    parts.push(local,bytes);central.push(record);offset+=local.length+bytes.length;
  }
  const centralSize=central.reduce((n,b)=>n+b.length,0),end=new Uint8Array(22),ev=new DataView(end.buffer);
  ev.setUint32(0,0x06054b50,true);ev.setUint16(8,entries.length,true);ev.setUint16(10,entries.length,true);ev.setUint32(12,centralSize,true);ev.setUint32(16,offset,true);
  return new Blob([...parts,...central,end],{type:'application/zip'});
}

export function listedResults(artifacts){
  const seen=new Set();return artifacts.filter(file=>{
    if(file.name.startsWith('mplconfig__'))return false;
    const key=JSON.stringify([file.name,file.sha256||file.data]);if(seen.has(key))return false;seen.add(key);return true;
  });
}

export async function buildAnalysisBundle(job,view=null){
  const results=listedResults(job.artifacts||[]),records=[...(job.files||[]).map(file=>({file,kind:'input'})),...results.map(file=>({file,kind:'result'}))];
  const estimate=records.reduce((n,{file})=>n+Math.ceil((file.data?.length||0)*3/4),0);
  if(estimate>limit)throw new Error('This dataset and its results exceed 512 MiB. Download individual results and keep your source datasets separately.');
  const entries=[],used=new Set(),manifest={schema:'spt.analysis-bundle.v1',analysisId:job.id,capturedAt:new Date().toISOString(),status:job.status,files:[]};
  for(const {file,kind} of records){
    if(typeof file.data!=='string')throw new Error('A file is not fully loaded. Wait for the analysis to load, then retry.');
    const bytes=decode(file.data),version=kind==='result'?artifactVersion(file,job.artifacts):null;
    const path=allocatePath(kind==='input'?'inputs':version.superseded?'earlier/response-'+version.response:'outputs',file.name,used);
    const sha256=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(n=>n.toString(16).padStart(2,'0')).join('');
    manifest.files.push({path,originalName:file.name,kind,bytes:bytes.length,sha256,...(version?{response:version.response,latest:!version.superseded}:{})});entries.push({path,bytes});
    await new Promise(resolve=>setTimeout(resolve,0));
  }
  const analysis={id:job.id,status:job.status,view,turns:(job.turns||[]).map(t=>({question:t.question,status:t.status,answer:t.answer,selection:t.selection}))};
  const readme='SPT data and results bundle\n\ninputs/ contains the original uploaded datasets.\noutputs/ contains the latest result files. Follow the analysis README and script instructions there; dependencies are not bundled.\nearlier/ contains superseded files, grouped by response. These are historical versions.\nmanifest.json maps original names to archive paths and records byte counts, SHA-256 checksums and response versions. Some filenames are adjusted for portable extraction.\nanalysis.json records discussion, submitted trajectory selections, and current view settings at export time. UMAP and t-SNE are stochastic; approximate PCA can sample points or dimensions. Recorded parameters do not guarantee the same layout when rerun.\n\nThis download preserves supplied files and discussion; it does not validate scientific inference.\n';
  entries.unshift({path:'manifest.json',bytes:encoder.encode(JSON.stringify(manifest,null,2)+'\n')},{path:'BUNDLE_README.txt',bytes:encoder.encode(readme)},{path:'analysis.json',bytes:encoder.encode(JSON.stringify(analysis,null,2)+'\n')});
  return {blob:zip(entries),filename:'spt-'+safeName(job.id)+'-data-results.zip',manifest};
}
