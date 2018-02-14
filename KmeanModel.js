dl = require('deeplearn');
d3 = require('d3');
class KmeanModel{
    constructor(points, k){
        this.math = dl.ENV.math;
        this.g = new dl.Graph();
        const a = dl.Array2D.new([points.length/2, 2], points);
        const b = dl.Array2D.new([3,2],[24,24,10,31,45,10]);
        this.points = new dl.Variable(a);
        this.center = new dl.Variable(b);
        this.k = k;
        this.learningRate = 0.1;
        this.inputTensor = null;
        this.costTensor = null;
        this.predictionTensor = null;
        this.feedEntries = null;
        this.optimizer = new dl.SGDOptimizer(this.learningRate);
        this.session = null;
        this.class = [];
    }
    setSession(n, k){
        var reCenters = this.math.reshape(this.math.tile(this.center,[n,1]),[n,k,2]);
        var rePoints = this.math.reshape(this.math.tile(this.points,[1,k]),[n,k,2]);
        var sumSquare = this.math.sum(this.math.square(this.math.sub(reCenters, rePoints)), 2);   
        var bestCenter = this.math.argMin(sumSquare, 1);
        this.class = bestCenter.dataSync();
        var points = this.points.dataSync();
        var next_Cluster = [0, 0, 0, 0, 0, 0];
        var times = [0,0,0];
        for(let i = 0; i < this.class.length; i++){
            times[this.class[i]] ++;
            next_Cluster[this.class[i]*2] = next_Cluster[this.class[i]*2] + points[i*2];
            
            next_Cluster[this.class[i]*2+1] = next_Cluster[this.class[i]*2 + 1] + points[i*2 + 1];
        }
        for(let i = 0; i < next_Cluster.length; i+=2){
            next_Cluster[i] = next_Cluster[i]/times[i/2];
            next_Cluster[i+1] = next_Cluster[i+1]/times[i/2];
        }
        const b = dl.Array2D.new([3,2],next_Cluster);
        this.center = new dl.Variable(b);
    }
    unpdate(){
        
    }
}
var width = 600;
var height = 600;
var padding = {left:30, right:30, top:20, bottom:80};
var svg = d3.select("body")				
.append("svg")				
.attr("width", width)		
.attr("height", height);	
function draw(model){
    var dataset = model.points.dataSync();
    var belong =  model.class;
    var xPoint = dataset.filter((val, index) => index % 2 == 0);
    var yPoint = dataset.filter((val, index) => index % 2 == 1);
    dataset = [];
    for(let i = 0; i < xPoint.length; i++)
        dataset.push([xPoint[i], yPoint[i], belong[i]]);
   
    var xScale  = d3.scaleLinear()
    .domain([0, d3.max(xPoint)]
    )
    .range([0, width - padding.left - padding.right]);
   
    
    var yScale  = d3.scaleLinear()
      .domain([0, 50])
      .range([height - padding.top - padding.bottom, 0]);

    var rectHeight = 25;	

    var rects = svg.selectAll(".MyRect")
    .data(dataset)
    .enter()
    .append('circle')
    .attr('class', 'point')
    .attr('cx', function(d) {
     return xScale(d[0])+32;
    })
    .attr('cy', function(d) {
     return yScale(d[1])+20;
    })
    .attr('r', 2)
    .style('fill', function(d, i){
        if(d[2] == 0)
            return 'rgb(20, 224, 238)';
        else if(d[2] == 1)
            return 'rgb(62, 121, 81)';
        else
            return 'rgb(80, 80, 80)';
    })
    .style('opacity', function(d, i){
       return 0.6;
    });

    var xAxis = d3.axisBottom(xScale).ticks(7);   
    var yAxis = d3.axisLeft(yScale).ticks(7); 
    svg.append("g")
    .attr("class","axis")
    .attr("transform","translate(" + padding.left + "," + (height - padding.bottom) + ")")
    .call(xAxis); 
          
  svg.append("g")
    .attr("class","axis")
    .attr("transform","translate(" + padding.left + "," + padding.top + ")")
    .call(yAxis);
}

const points = [];
var numPoint = 4000;
for(let i = 0; i < numPoint; i++){
    points.push(Math.random()*50);
    points.push(Math.random()*50);
}
var step = 0;
model = new KmeanModel(points, 2);
function kMean(){
    if(step > 20)
        return ;
    model.math.scope(() => {
        requestAnimationFrame(kMean);
        model.setSession(numPoint, 3);
        draw(model);
        step++;
        }
    )
   
}

kMean();

