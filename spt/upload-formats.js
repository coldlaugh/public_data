const signatures={
 pdf:b=>starts(b,[37,80,68,70,45]),
 png:b=>starts(b,[137,80,78,71,13,10,26,10]),
 jpg:b=>starts(b,[255,216,255]),
 jpeg:b=>starts(b,[255,216,255]),
 webp:b=>starts(b,[82,73,70,70])&&starts(b.slice(8),[87,69,66,80]),
 gif:b=>starts(b,[71,73,70,56,55,97])||starts(b,[71,73,70,56,57,97]),
 tif:b=>starts(b,[73,73,42,0])||starts(b,[77,77,0,42])||starts(b,[73,73,43,0])||starts(b,[77,77,0,43])
};
signatures.tiff=signatures.tif;
const starts=(bytes,prefix)=>prefix.every((value,i)=>bytes[i]===value);
export function uploadKind(name){if(!name.includes('.'))return null;const ext=name.split('.').pop().toLowerCase();return Object.hasOwn(signatures,ext)?ext:null;}
export function validateUpload(name,bytes){
 const kind=uploadKind(name);
 if(kind){if(!signatures[kind](bytes))throw new Error(name+': file contents do not match the supported PDF/image extension.');return kind;}
 let text;try{text=new TextDecoder('utf-8',{fatal:true}).decode(bytes);}catch{throw new Error(name+': upload UTF-8 text, PDF, PNG, JPEG, WebP, GIF or TIFF; other binary formats are unsupported.');}
 if(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/.test(text))throw new Error(name+': text contains unsupported binary control characters.');
 return 'text';
}
