export const config = {
  runtime: 'edge',
};

export default async function handler(request) {
  // We only want to handle POST requests from our frontend
  if (request.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405 });
  }

  try {
    // Get the request body (which includes the image and prompt) from the frontend
    const requestBody = await request.json();

    // Safely get the API key from the Vercel environment variables you already set up
    const apiKey = process.env.VITE_GEMINI_API_KEY;

    // If the API key isn't set on Vercel, return an error
    if (!apiKey) {
      return new Response('API key is not configured on the server.', { status: 500 });
    }

    const googleApiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=${apiKey}`;

    // Forward the exact same request from our frontend to the Google API, but with the key added securely
    const googleApiResponse = await fetch(googleApiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    // If Google returns an error, pass it along to our frontend for debugging
    if (!googleApiResponse.ok) {
      const errorText = await googleApiResponse.text();
      console.error("Google API Error:", errorText);
      return new Response(errorText, { status: googleApiResponse.status });
    }

    // If successful, send Google's response back to our frontend
    return new Response(googleApiResponse.body, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

  } catch (error) {
    console.error('Proxy Error:', error);
    return new Response('An internal server error occurred in the proxy.', { status: 500 });
  }
}