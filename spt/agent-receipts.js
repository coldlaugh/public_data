// Tab-scoped retry metadata only: no questions, selected IDs, source bytes or tokens.
const KEY='spt.agent.receipts.v1',MAX_BYTES=65536;
const digest=value=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
function clean(record){
 if(!record||!digest(record.key)||!digest(record.signature))return null;
 const submission=record.submission;
 if(submission!==null&&(!submission||typeof submission.id!=='string'||submission.id.length>128||typeof submission.status!=='string'||submission.status.length>64))return null;
 return {key:record.key,signature:record.signature,submission:submission?{id:submission.id,status:submission.status}:null};
}
export function createAgentReceipts(storage){
 const read=()=>{try{const raw=storage?.getItem(KEY);if(!raw||raw.length>MAX_BYTES)return [];const parsed=JSON.parse(raw);return Array.isArray(parsed)?parsed.slice(-128).map(clean).filter(Boolean):[];}catch{return [];}};
 return {
  get(key){return read().find(record=>record.key===key)||null;},
  save(key,signature,submission=null){
   try{if(!storage)return false;const record=clean({key,signature,submission});if(!record)return false;
    let records=read().filter(record=>record.key!==key);if(records.length>=128)return false;records.push(record);
    const raw=JSON.stringify(records);if(raw.length>MAX_BYTES)return false;storage.setItem(KEY,raw);return true;
   }catch{return false;}
  }
 };
}
