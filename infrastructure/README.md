# Infrastructure Templates (Showcase)

This folder contains infrastructure-as-code templates for a scalable architecture, as described in the main project README. **All files are for demonstration only and are fully commented out.**

## Included Templates

- **aws-dynamo-template.yaml**: AWS CloudFormation template for DynamoDB
- **cloudflare-cdn-template.txt**: Cloudflare CDN configuration example
- **nginx-loadbalancer.conf**: NGINX load balancer config
- **k8s-api-deployment.yaml**: Kubernetes deployment for API layer
- **redis-cache-template.yaml**: Kubernetes deployment for Redis cache
- **aws-api-gateway-template.yaml**: AWS CloudFormation template for API Gateway

These templates are intended to illustrate how you might set up scalable, cloud-native infrastructure for a production-grade system capable of handling 10,000+ concurrent users and 1,000+ API calls per minute.

**Note:**
- All resources, names, and settings are placeholders.
- Uncomment and modify as needed for real deployments. 