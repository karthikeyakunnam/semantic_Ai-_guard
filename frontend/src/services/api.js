const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
async function request(path, options={}) { const res=await fetch(`${API}${path}`,options); if(!res.ok){let message=`Request failed (${res.status})`;try{message=(await res.json()).detail||message}catch{} throw new Error(message)} return res; }
export async function getDocuments(){return (await request('/api/documents')).json()}
export async function uploadDocument(file){const body=new FormData();body.append('file',file);return (await request('/api/documents',{method:'POST',body})).json()}
export async function compareDocuments(document_a_id,document_b_id){return (await request('/api/analysis/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({document_a_id,document_b_id})})).json()}
export async function getReports(){return (await request('/api/reports')).json()}
export async function downloadReport(id){const res=await request(`/api/reports/${id}/download`);const url=URL.createObjectURL(await res.blob());const a=document.createElement('a');a.href=url;a.download=`semanticguard-report-${id}.pdf`;a.click();URL.revokeObjectURL(url)}
export async function deleteDocument(id){await request(`/api/documents/${id}`,{method:'DELETE'})}
