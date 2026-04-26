# Corporate TLS Certificates

If your machine sits behind a TLS-inspecting proxy (Zscaler, Netskope,
Palo Alto, etc.), every outbound HTTPS call from this stack will fail with:

```
SSLError: CERTIFICATE_VERIFY_FAILED
```

This is because the proxy substitutes its own certificate, which the public
CAs in your system trust store don't recognize. The fix is to point the
stack at your company's CA bundle.

## Quick setup

Drop your CA bundle at the repo root under `cert/knapp.pem`:

```
RAG_Bench/
├── cert/
│   └── knapp.pem        # your corporate CA bundle (gitignored)
├── .env
└── ...
```

That's it. Both `rag_clients.load_env()` (used by every builder and server)
and `openai_llm.load_env_file()` (used by the Search-o1 client scripts)
look for this file at startup.

When the bundle is found you'll see one log line on startup:

```
[ca-bundle] Using company CA bundle: /abs/path/to/cert/knapp.pem
```

If `cert/knapp.pem` is missing, the helper is a silent no-op — your system
default CA store is used. This is the right behavior on machines that
aren't behind a corporate proxy.

## What happens under the hood

When a bundle is found, the stack:

1. Sets `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE` in the process environment
   so `requests` and `httpx` honor it transparently.
2. Passes `verify=<bundle>` explicitly to the OpenAI SDK's underlying
   `httpx.Client`.

This means every HTTP call — embeddings, reranker, LLM, retriever-to-retriever
— picks up the bundle without any per-client configuration.

## Custom path

If your bundle lives somewhere else, set `COMPANY_CA_CERT` in `.env`
(or in the shell environment):

```bash
COMPANY_CA_CERT=/etc/ssl/certs/my-company-ca.pem
```

The override always wins over `cert/knapp.pem`.

## Why `cert/` is gitignored

CA bundles are environment-specific. Committing one would push proxy
internals to GitHub and break setups on other networks. Keep it local.
