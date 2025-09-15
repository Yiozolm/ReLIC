## ReLIC: Residual flow matching for Learned Image Compression

Official Pytorch implementation for paper 'ReLIC: Residual flowmatching for Learned Image Compression'.

First attempt to incorporate Flow Matching (FM) into the learned image compression framework.

### QuickStart 

---

**Training**

Train on [OpenImage dataset](https://storage.googleapis.com/openimages/web/index.html)(400k images).

We offer `train.sh` script.


**Supported FM models**

- from `Torchcfm`

- [X] ConditionalFlowMatcher
- [X] ExactOptimalTransportConditionalFlowMatcher
- [X] TargetConditionalFlowMatcher
- [X] SchrodingerBridgeConditionalFlowMatcher
- [X] VariancePreservingConditionalFlowMatcher

- Others (Not tested in Relic)

- [X] Meanflow
- [X] RectifiedFlowMatcher
- [X] GeoOptManifoldFlowMatcher
- [X] ExactOptimalTransportGeoOptManifoldFlowMatcher


### Acknoewledgement

- [Compressai](https://github.com/InterDigitalInc/CompressAI): A PyTorch library and evaluation platform for end-to-end compression research.
- [Torchcfm](https://github.com/atong01/conditional-flow-matching):  A Conditional Flow Matching library
