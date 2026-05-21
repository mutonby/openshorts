import SaaShortsTab from '../../components/SaaShortsTab';
import { useKeys } from '../../state/keysStore.js';

export default function LegacySaaSShorts() {
  const keys = useKeys();
  return (
    <SaaShortsTab
      geminiApiKey={keys.gemini}
      elevenLabsKey={keys.elevenLabs}
      falKey={keys.fal}
      uploadPostKey={keys.uploadPost}
      uploadUserId={keys.uploadUserId}
    />
  );
}
