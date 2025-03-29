# setup mosek and download the required packages
mkdir -p ~/mosek
cp mosek.lic ~/mosek

# get input for gemini_api_key
read -p "Enter your GEMINI_API_KEY : " gemini_api_key

# save the gemini_api_key to Gemini_Inference/.env
echo "GEMINI_API_KEY=$gemini_api_key" > Gemini_Inference/.env