// Read quoted delimited text without evaluating or changing the uploaded file.
export function delimitedRows(text,separator=','){
  const rows=[];let row=[],field='',quoted=false,closed=false;
  const endField=()=>{row.push(field);field='';closed=false;};
  const endRow=()=>{endField();if(row.some(value=>value.trim()))rows.push(row);row=[];};
  text=text.replace(/^\uFEFF/,'');
  for(let i=0;i<text.length;i++){
    const char=text[i];
    if(quoted){
      if(char==='"'){if(text[i+1]==='"'){field+='"';i++;}else{quoted=false;closed=true;}}
      else field+=char;
    }else if(char===separator)endField();
    else if(char==='\n'||char==='\r'){endRow();if(char==='\r'&&text[i+1]==='\n')i++;}
    else if(char==='"'){
      if(closed||field.trim())throw new Error('Invalid quoted field.');
      field='';quoted=true;
    }else if(closed){if(!/\s/.test(char))throw new Error('Unexpected text after a quoted field.');}
    else field+=char;
  }
  if(quoted)throw new Error('Unclosed quoted field.');
  if(field||row.length||closed)endRow();
  return rows;
}

const numeric=value=>{
  const text=value?.trim();
  if(!text||!/^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/.test(text))return null;
  const number=Number(text);return Number.isFinite(number)?number:null;
};

export function parseTrajectoryTable(text,filename){
  const timing={duplicateTimes:0,duplicateTimeTracks:0,reorderedTracks:0};
  const rows=delimitedRows(text,/\.tsv$/i.test(filename)?'\t':',');
  if(!rows.length)return {tracks:[],skippedRows:0,...timing};
  const columns=rows[0].map(value=>value.trim());
  const ix=columns.indexOf('x_um'),iy=columns.indexOf('y_um'),id=columns.indexOf('track_id'),it=columns.indexOf('t_s');
  if(ix<0||iy<0||id<0)return {tracks:[],skippedRows:0,...timing};
  if(['x_um','y_um','track_id','t_s'].some(name=>columns.filter(value=>value===name).length>1))throw new Error('Duplicate trajectory columns.');
  const byId=new Map();let skippedRows=0;
  for(const row of rows.slice(1)){
    const x=numeric(row[ix]),y=numeric(row[iy]),time=it<0?null:numeric(row[it]),trackId=row[id]?.trim();
    if(!trackId||x===null||y===null||it>=0&&time===null){skippedRows++;continue;}
    if(!byId.has(trackId))byId.set(trackId,[]);
    byId.get(trackId).push({x,y,time});
  }
  const tracks=[...byId].map(([id,observations])=>{
    if(it>=0){
      const seen=new Set();let duplicates=0,reordered=false;
      for(let i=0;i<observations.length;i++){
        const time=observations[i].time;
        if(seen.has(time))duplicates++;else seen.add(time);
        if(i&&time<observations[i-1].time)reordered=true;
      }
      timing.duplicateTimes+=duplicates;
      if(duplicates)timing.duplicateTimeTracks++;
      if(reordered)timing.reorderedTracks++;
      observations.sort((a,b)=>a.time-b.time);
    }
    return {id,key:filename+':'+id,file:filename,path:observations.map(o=>[o.x,o.y]),times:it<0?[]:observations.map(o=>o.time)};
  });
  return {tracks,skippedRows,...timing};
}

// Keep the control bounded without making any tracks inaccessible to a refined search.
export function matchingTracks(tracks,query,limit=200){
  const search=query.trim().toLocaleLowerCase(),matches=tracks.filter(track=>track.key.toLocaleLowerCase().includes(search));
  return {tracks:matches.slice(0,limit),total:matches.length};
}

export function rawPreviewTracks(parsed,previous,keepSelection){
  const byId=new Map(parsed.map(track=>[track.key,track]));
  if(keepSelection)for(const track of previous)if(track.fromEncoder)byId.set(track.key,track);
  return [...byId.values()];
}

export function encoderPreviewTrack(point,previous){
  const path=point.xPosition.map((x,i)=>[x,point.yPosition[i]]);
  const sameObservations=previous?.times?.length===path.length&&previous.path.length===path.length&&previous.path.every(([x,y],i)=>x===path[i][0]&&y===path[i][1]);
  return {id:point.trackId,key:point.id,file:point.file,path,times:sameObservations?previous.times:[],fromEncoder:true};
}
