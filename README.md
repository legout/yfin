# yFin - A blazing fast python wrapper for the Yahoo Fincance API

Simple python wrapper for the yahoo finance api. Heavily uses asyncronous requests to speed up data downloads. Optional usage of random proxies is possible.

This python library currently has functions to download
 - historical OHLCV data
 - fundamental data
 - quotes
 - symbol search


## Installation
```
pip install git+https://github.com/legout/yfin.git
```

## Exapmples

#

## Use of a random proxy server.

**Note** The library should work fine without using random proxies, but the number of requests is limited to ~1000/h. Using random proxies might be illegal.

You can set `random-proxy=True` in any of the yFin functions. By default this uses free proxies*. In my experience, these proxies are not reliable, but maybe you are lucky.

### Webshare.io proxies
I am using proxies from [webshare.io](https://www.webshare.io/). I am very happy with their service and the pricing. If you wanna use their service too, sign up (use the [this link](https://www.webshare.io/?referral_code=upb7xtsy39kl) if you wanna support my work) and choose a plan that fits your needs. In the next step, go to Dashboard -> Proxy -> List -> Download and copy the download link. Set this download link as an environment variable `WEBSHARE_PROXIES_URL`  before importing any yFin function. 

*Export WEBSHARE_PROXIES_URL in your linux shell*
```
$ export WEBSHARE_PROXIES_URL="https://proxy.webshare.io/api/v2/proxy/list/download/abcdefg1234567/-/any/username/direct/-/"
```

You can also set this environment variable permanently in an `.env` file (see the `.env-exmaple`) in your home folder or current folder or in your command line config file (e.g. `~/.bashrc`).

*Write WEBSHARE_PROXIES_URL into .env*
```
WEBSHARE_PROXIES_URL="https://proxy.webshare.io/api/v2/proxy/list/download/abcdefg1234567/-/any/username/direct/-/"
```

*or write WEBSHARE_PROXIES_URL into your shell config file (e.g. ~/.bashrc)*
```
$ echo 'export WEBSHARE_PROXIES_URL="https://proxy.webshare.io/api/v2/proxy/list/download/abcdefg1234567/-/any/username/direct/-/"' >> ~/.bashrc
```


*Free Proxies are scraped from here:
- "http://www.free-proxy-list.net"
- "https://free-proxy-list.net/anonymous-proxy.html"
- "https://www.us-proxy.org/"
- "https://free-proxy-list.net/uk-proxy.html"
- "https://www.sslproxies.org/"


<hr>

#### Support my work :-)

If you find this useful, you can buy me a coffee. Thanks!

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/W7W0ACJPB)

