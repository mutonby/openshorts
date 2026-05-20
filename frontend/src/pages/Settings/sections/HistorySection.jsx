import PlaceholderSection from './PlaceholderSection.jsx';

export default function HistorySection() {
  return (
    <PlaceholderSection
      title="Processing history"
      description="Past clip-generator jobs, short-form batches, and long-form renders. Re-download outputs or re-run a job with new settings."
      todo={[
        'Searchable list of past jobs',
        'Per-job re-download links',
        'Re-edit a clip in a new wizard run',
        'Backend index endpoint (plan TODO #10)',
      ]}
    />
  );
}
