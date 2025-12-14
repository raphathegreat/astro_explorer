# 🚀 AstroPi Explorer Dashboard - Deployment Guide

This guide provides the **easiest ways** to deploy your AstroPi Explorer Dashboard to the internet.

## 🎯 **Recommended Deployment Options (Easiest to Hardest)**

### 1. 🥇 **Railway** (Easiest - Recommended)
**Why Railway?** Free tier, automatic deployments, built-in database, simple setup.

#### Quick Setup:
1. **Sign up** at [railway.app](https://railway.app)
2. **Connect GitHub** and select your repository
3. **Deploy automatically** - Railway detects Flask apps
4. **Add environment variables** (optional):
   - `FLASK_ENV=production`
   - `PORT=5000` (Railway sets this automatically)

**Cost:** Free tier includes 500 hours/month
**Time to deploy:** 5-10 minutes

---

### 2. 🥈 **Render** (Very Easy)
**Why Render?** Free tier, automatic SSL, simple configuration.

#### Quick Setup:
1. **Sign up** at [render.com](https://render.com)
2. **Create New Web Service**
3. **Connect GitHub** repository
4. **Configure:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn iss_speed_html_dashboard_v2_clean:app --bind 0.0.0.0:$PORT`
   - Environment: `Python 3`

**Cost:** Free tier with 750 hours/month
**Time to deploy:** 10-15 minutes

---

### 3. 🥉 **Heroku** (Easy but requires credit card)
**Why Heroku?** Most popular, extensive documentation.

#### Quick Setup:
1. **Install Heroku CLI** and sign up
2. **Create Heroku app:**
   ```bash
   heroku create your-astro-dashboard
   ```
3. **Deploy:**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

**Cost:** $7/month (no free tier anymore)
**Time to deploy:** 15-20 minutes

---

## 📋 **Pre-Deployment Checklist**

### ✅ **Files Already Created:**
- ✅ `requirements.txt` - Production dependencies
- ✅ `Procfile` - Heroku/Railway deployment config
- ✅ `env.example` - Environment variables template

### ✅ **Required Actions:**

1. **Test locally first:**
   ```bash
   pip install -r requirements.txt
   python iss_speed_html_dashboard_v2_clean.py
   ```

2. **Prepare your images:**
   - Ensure `photos-1/`, `photos-2/`, etc. folders are in your repository
   - Images will be deployed with your app

3. **Commit all files:**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

---

## 🚀 **Step-by-Step: Railway Deployment (Recommended)**

### Step 1: Prepare Repository
```bash
# Ensure all files are committed
git add .
git commit -m "Add deployment files"
git push origin main
```

### Step 2: Deploy on Railway
1. Go to [railway.app](https://railway.app)
2. Click **"Start a New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your AstroPi Explorer repository
5. Railway will automatically detect it's a Flask app
6. Click **"Deploy"**

### Step 3: Configure (Optional)
1. Go to your project dashboard
2. Click **"Variables"** tab
3. Add environment variables:
   - `FLASK_ENV=production`
   - `FLASK_DEBUG=False`

### Step 4: Access Your App
- Railway provides a URL like: `https://your-app-name.railway.app`
- Your dashboard will be live at this URL!

---

## 🚀 **Step-by-Step: Render Deployment**

### Step 1: Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with GitHub

### Step 2: Create Web Service
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Configure:
   - **Name:** `astro-dashboard`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn iss_speed_html_dashboard_v2_clean:app --bind 0.0.0.0:$PORT`

### Step 3: Deploy
1. Click **"Create Web Service"**
2. Render will build and deploy automatically
3. Get your URL: `https://astro-dashboard.onrender.com`

---

## 🔧 **Production Configuration**

### Environment Variables
Create a `.env` file (copy from `env.example`):
```bash
FLASK_ENV=production
FLASK_DEBUG=False
PORT=5000
HOST=0.0.0.0
```

### Production Dependencies
The `requirements.txt` includes:
- Flask 2.0+ (web framework)
- OpenCV (computer vision)
- NumPy (numerical processing)
- Matplotlib (visualization)
- Gunicorn (production WSGI server)

---

## 🐛 **Troubleshooting**

### Common Issues:

**Q: App won't start?**
- Check that `Procfile` exists and is correct
- Ensure `requirements.txt` has all dependencies
- Check build logs in your deployment platform

**Q: Images not loading?**
- Ensure `photos-*` folders are committed to git
- Check file paths in the application

**Q: Slow performance?**
- Consider upgrading to paid tier for better resources
- Optimize image sizes if needed

**Q: Memory issues?**
- Reduce `max_features` in the dashboard settings
- Use ORB instead of SIFT algorithm

---

## 💰 **Cost Comparison**

| Platform | Free Tier | Paid Tier | Best For |
|----------|-----------|-----------|----------|
| **Railway** | 500 hours/month | $5/month | Easiest setup |
| **Render** | 750 hours/month | $7/month | Good free tier |
| **Heroku** | None | $7/month | Most popular |

---

## 🎉 **After Deployment**

Once deployed, your AstroPi Explorer Dashboard will be available at:
- **Railway:** `https://your-app-name.railway.app`
- **Render:** `https://your-app-name.onrender.com`
- **Heroku:** `https://your-app-name.herokuapp.com`

### Features Available Online:
- ✅ Real-time ISS speed analysis
- ✅ Interactive filtering system
- ✅ Cloudiness classification
- ✅ Statistical visualizations
- ✅ All 637 images from your dataset

---

## 🔄 **Updating Your Deployment**

To update your deployed app:
```bash
# Make your changes
git add .
git commit -m "Update dashboard"
git push origin main

# Most platforms auto-deploy on git push
# Check your platform's dashboard for deployment status
```

---

**🎯 Recommendation: Start with Railway for the easiest deployment experience!**

*Need help? Check the platform-specific documentation or create an issue in your repository.*

