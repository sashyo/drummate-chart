export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname.startsWith('/api/')) {
      if (!env.BACKEND_ORIGIN) {
        return new Response(
          JSON.stringify({ detail: 'The transcription backend is not deployed yet.' }),
          { status: 503, headers: { 'content-type': 'application/json' } });
      }
      const origin = new URL(env.BACKEND_ORIGIN);
      url.protocol = origin.protocol;
      url.host = origin.host;
      // Transcriptions run for minutes; stream the proxied response through.
      return fetch(new Request(url, request));
    }
    return env.ASSETS.fetch(request);
  },
};
