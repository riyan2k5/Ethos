# Setting Up GitHub Secrets for Database Access

This guide explains how to securely store your database credentials in GitHub Secrets so that GitHub Actions can access them without exposing them publicly.

## Step 1: Add Database URL to GitHub Secrets

1. Go to your GitHub repository: `https://github.com/riyan2k5/Ethos`
2. Click on **Settings** (in the repository, not your account settings)
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Set the following:
   - **Name**: `DATABASE_URL`
   - **Value**: Your full PostgreSQL connection string
     ```
     postgresql://username:password@host:port/database?sslmode=require&channel_binding=require
     ```
     Example format for Neon:
     ```
     postgresql://neondb_owner:your_password@ep-xxx-pooler.region.aws.neon.tech/neondb?sslmode=require&channel_binding=require
     ```
6. Click **Add secret**

## Step 2: Verify Secret is Set

After adding the secret, you should see `DATABASE_URL` listed in your repository secrets. The value will be hidden (shown as `••••••••`).

## Step 3: How It Works

- The GitHub Actions workflow (`model-training.yml`) reads the secret using `${{ secrets.DATABASE_URL }}`
- The secret is injected as an environment variable during the workflow run
- The secret is **never** exposed in logs or code
- Only repository collaborators with appropriate permissions can view/edit secrets

## Step 4: Local Development

For local development, create a `.env` file in the project root:

```bash
cp env.template .env
```

Then edit `.env` and add your database connection string:

```
DATABASE_URL=postgresql://username:password@host:port/database?sslmode=require&channel_binding=require
```

Replace `username`, `password`, `host`, `port`, and `database` with your actual Neon database credentials.

**Important**: The `.env` file is already in `.gitignore` and will never be committed to git.

## Security Best Practices

✅ **DO:**
- Store sensitive credentials in GitHub Secrets
- Use `.env` files for local development (already in `.gitignore`)
- Rotate credentials periodically
- Use different credentials for development and production if possible

❌ **DON'T:**
- Commit `.env` files to git
- Hardcode credentials in code
- Share credentials in pull requests or issues
- Use production credentials in local development

## Troubleshooting

### Workflow fails with "DATABASE_URL secret is not set"
- Make sure you've added the secret in the correct repository
- Check that the secret name is exactly `DATABASE_URL` (case-sensitive)
- Verify you have permission to view repository secrets

### Connection fails in GitHub Actions
- Verify the connection string is correct
- Check that your database allows connections from GitHub Actions IPs
- Ensure SSL mode is set correctly for Neon databases

