import ThumbnailStudio from '../../components/ThumbnailStudio';
import { useKeys } from '../../state/keysStore.js';

export default function LegacyThumbnails() {
  const keys = useKeys();
  return (
    <ThumbnailStudio
      geminiApiKey={keys.gemini}
      uploadPostKey={keys.uploadPost}
      uploadUserId={keys.uploadUserId}
    />
  );
}
