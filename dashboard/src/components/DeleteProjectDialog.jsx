import { useRef, useState } from 'react';
import { Loader2, Trash2 } from 'lucide-react';
import { apiJson } from '../lib/api';
import Modal from './ui/Modal';

export default function DeleteProjectDialog({ project, onClose, onDeleted }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const sending = useRef(false);

  const remove = async () => {
    if (sending.current) return;
    sending.current = true;
    setBusy(true);
    setError('');
    try {
      await apiJson(`/api/projects/${encodeURIComponent(project.jobId)}`, { method: 'DELETE' });
    } catch (err) {
      setError(err.detail || 'Could not delete this project. Try again.');
      setBusy(false);
      sending.current = false;
      return;
    }
    onDeleted(project.jobId);
  };

  return (
    <Modal
      isOpen
      title="delete project"
      onClose={busy ? undefined : onClose}
      footer={
        <div className="flex flex-wrap justify-end gap-2">
          <button type="button" className="btn-quiet" onClick={onClose} disabled={busy}>keep project</button>
          <button type="button" className="btn-danger" onClick={remove} disabled={busy}>
            {busy ? <Loader2 size={15} className="animate-spin" /> : <Trash2 size={15} />}
            {busy ? 'deleting…' : 'permanently delete'}
          </button>
        </div>
      }
    >
      {project.title && <p className="mb-2 break-words font-medium text-ink">{project.title}</p>}
      <p className="text-sm text-muted">Permanently delete this project, including its saved clips and source files.</p>
      {error && <p className="mt-3 text-sm text-danger" role="alert">{error}</p>}
    </Modal>
  );
}
