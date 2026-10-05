// Tab-scoped view preferences only: never serialize uploads, questions or tokens.
const KEY='spt.analysis.views.v1',MAX_RECORD=256*1024,MAX_TOTAL=1024*1024;
const parameterKeys=['visibleTrackIds','rotationEnabled','neighborCount','distanceMetric','umapIs3d','umapNeighbors','umapMinDist','pcaIs3d','pcaX','pcaY','pcaZ','tSNEis3d','tsnePerplexity','tsneLearningRateExponent','tsneSuperviseFactor'];
function cleanView(value){
  if(!value||!['raw','pca','umap','tsne'].includes(value.projection)||!Array.isArray(value.selection)||value.selection.some(id=>typeof id!=='string'))return null;
  const parameters=value.projectionSettings;
  if(parameters!==null&&parameters!==undefined&&(typeof parameters!=='object'||Array.isArray(parameters)))return null;
  const validParameter=(key,v)=>key==='visibleTrackIds'?v===null||Array.isArray(v)&&v.every(id=>typeof id==='string'):key==='distanceMetric'?['cosine','euclidean'].includes(v):['rotationEnabled','umapIs3d','pcaIs3d','tSNEis3d'].includes(key)?typeof v==='boolean':Number.isFinite(v);
  return {projection:value.projection,selection:[...new Set(value.selection)],projectionSettings:parameters?Object.fromEntries(parameterKeys.filter(key=>Object.hasOwn(parameters,key)&&validParameter(key,parameters[key])).map(key=>[key,parameters[key]])):null};
}
export function createViewRecovery(storage){
  const read=()=>{const raw=storage.getItem(KEY);if(!raw||raw.length>MAX_TOTAL)return [];const records=JSON.parse(raw);return Array.isArray(records)?records.filter(r=>typeof r?.id==='string'&&cleanView(r.view)):[];};
  return {
    load(id){try{return cleanView(read().find(r=>r.id===id)?.view);}catch{return null;}},
    save(id,value){
      try{
        let records=read().filter(r=>r.id!==id),view=cleanView(value);
        if(typeof id!=='string'||!view||JSON.stringify(view).length>MAX_RECORD){storage.setItem(KEY,JSON.stringify(records));return false;}
        records.push({id,view});records=records.slice(-12);
        while(JSON.stringify(records).length>MAX_TOTAL)records.shift();
        storage.setItem(KEY,JSON.stringify(records));return true;
      }catch{
        // Avoid restoring an older preference after a failed update.
        try{storage.removeItem(KEY);}catch{}
        return false;
      }
    },
  };
}
