mapboxgl.accessToken = 'pk.eyJ1IjoiYWlzaGNoYW1hcnRoaSIsImEiOiJjbHB1Yjk2djcwajBlMmluenJvdGlucG54In0.1nBG1ilIoMJlD1xJ4mzIoA';
const map = new mapboxgl.Map({
  container: 'map',
  style: 'mapbox://styles/mapbox/streets-v12',
  center: [78.4736, 17.3855],
  zoom: 16
});

map.on('load', () => {
  map.addSource('places', {
    'type': 'geojson',
    'data': {
      'type': 'FeatureCollection',
      'features': [
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:01 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am <br> Cordinates : [78.4713, 17.3885] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4710, 17.3890]
          }
        },

        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:02 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56AmCordinates : [78.4713, 17.3885] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4713, 17.3885]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:03 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am Cordinates : [78.4717, 17.3880] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4717, 17.3880]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:04 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am Cordinates : [78.4721, 17.3875] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4721, 17.3875]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:05 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am Cordinates : [78.4725, 17.3870] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4725, 17.3870]
          }
        },

        // red line ....
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:06 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am Cordinates : [78.4720, 17.3866] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4720, 17.3866]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:07 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56AmCordinates : [78.4716, 17.3864] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4716, 17.3864]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:08 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am Cordinates : [78.4719, 17.3858] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4719, 17.3858]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:09 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am Cordinates : [78.4722, 17.3852] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4722, 17.3852]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:10 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am Cordinates : [78.4725, 17.3846] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4725, 17.3846]
          }
        },

        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:11 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am Cordinates : [78.4728, 17.3840] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4728, 17.3840]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:12 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am Cordinates : [78.4731, 17.3834] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4731, 17.3834]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:13 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56AmCordinates : [78.4722, 17.3852] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4722, 17.3852]
          }
        },

        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:14 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56AmCordinates : [78.4729, 17.3865] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4729, 17.3865]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:15 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am Cordinates : [78.4732, 17.3860] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4732, 17.3860]
          }
        },


        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:16 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am Cordinates :  [78.4736, 17.3855] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4736, 17.3855]
          }
        },

        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:17 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am Cordinates : [78.4740 , 17.3850]</strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4740, 17.3850]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:18 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am Cordinates : [78.4744 , 17.3845] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4744, 17.3845]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:19 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56AmCordinates : [78.4748 , 17.3840]</strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4748, 17.3840]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:20 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am Cordinates : [78.4752 , 17.3835]</strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4752, 17.3835]
          }
        },

        // bottom left
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:21 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56AmCordinates : [78.4742 , 17.3839]</strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4742, 17.3839]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>LocationId:22 <br>Status: operational <br> temperature:25.79 Celsius <br> Averge Dump Weight : 10.15 kgs <br>   LastUpdated: 2024-02-20::9:56Am Cordinates : [78.4736 , 17.3835] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4736, 17.3835]
          }
        },

        // bottom right 
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4756 , 17.38441] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4756, 17.38441]
          }
        },

        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :78.4760 , 17.3849] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4760, 17.3849]
          }
        },

        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4761 , 17.3857] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4761, 17.3857]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4762 , 17.3864] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4762, 17.3864]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4762 , 17.3869] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4762, 17.3869]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.47625 , 17.3876] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.47625, 17.3876]
          }
        },

        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4763 , 17.3883] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4763, 17.3883]
          }
        },

        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4760 , 17.3886] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4760, 17.3886]
          }
        },

        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4754 , 17.3886] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4754, 17.3886]
          }
        },

        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4748 , 17.3886 ] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4748, 17.3886]
          }
        },

        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4742 , 17.38868 ] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4742, 17.38868]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4742 , 17.38868 ] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4736, 17.38893]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4742 , 17.38868 ] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4730, 17.38935]
          }
        },

        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4742 , 17.38868 ] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4730, 17.38935]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4742 , 17.38868 ] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4730, 17.38935]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4742 , 17.38868 ] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4730, 17.38935]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4742 , 17.38868 ] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4730, 17.38935]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4742 , 17.38868 ] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4729, 17.389]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4742 , 17.38868 ] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4727, 17.3883]
          }
        },
        {
          'type': 'Feature',
          'properties': {
            'description': '<strong>SensorId:19 <br>Status: operational <br> pressure:65 psi  <br> temperature:25.8 Celsius <br> flowRate: 101.5 gallons/min <br> timestamp": 2023-12-19::9:56 Cordinates :[78.4742 , 17.38868 ] </strong><p>  </p>'
          },
          'geometry': {
            'type': 'Point',
            'coordinates': [78.4724, 17.3878]
          }
        },

      ]
    }
  });

  map.addLayer({
    'id': 'places',
    'type': 'circle',
    'source': 'places',
    'paint': {
      'circle-color': '#4264fb',
      'circle-radius': 6,
      'circle-stroke-width': 2,
      'circle-stroke-color': '#ffffff'
    }
  });

  const lineStringCoordinates = [
    [78.4710, 17.3890], // Start marker coordinates
    [78.4713, 17.3885],// Additional marker coordinates
    [78.4717, 17.3880], // Additional marker coordinates
    [78.4725, 17.3870],
    [78.4729, 17.3865],
    [78.4734, 17.3857],
    [78.4738, 17.3853],
    [78.4736, 17.3855],
    [78.4740, 17.3850],
    [78.4744, 17.3845],
    [78.4748, 17.3840],
    [78.4736, 17.3835],
    [78.4752, 17.3835],
    [78.4742, 17.3839],


    // Additional marker coordinates
    // Add all marker coordinates as needed
    // ...
  ];

  map.addSource('line', {
    type: 'geojson',
    data: {
      type: 'Feature',
      geometry: {
        type: 'LineString',
        coordinates: lineStringCoordinates
      }
    }
  });

  map.addLayer({
    id: 'line',
    type: 'line',
    source: 'line',
    paint: {
      'line-color': 'blue',
      'line-width': 2
    }
  });

  const redLineCoordinates = [
    [78.4725, 17.3870],// Start of the red line coordinates
    [78.4720, 17.3866], // Additional coordinates for the red line
    [78.4716, 17.3864],
    [78.4719, 17.3858],
    [78.4722, 17.3852],
    [78.4725, 17.3846],
    [78.4728, 17.3840],
    [78.4731, 17.3834],
    [78.4722, 17.3852],


    // Additional coordinates for the red line
    // Add all coordinates for the red line as needed
    // ...
  ];

  map.addSource('redLine', {
    type: 'geojson',
    data: {
      type: 'Feature',
      geometry: {
        type: 'LineString',
        coordinates: redLineCoordinates
      }
    }
  });

  map.addLayer({
    id: 'redLine',
    type: 'line',
    source: 'redLine',
    paint: {
      'line-color': 'red',
      'line-width': 2
    }
  });




  const popup = new mapboxgl.Popup({
    closeButton: false,
    closeOnClick: false
  });

  map.on('mouseenter', 'places', (e) => {
    map.getCanvas().style.cursor = 'pointer';
    const coordinates = e.features[0].geometry.coordinates.slice();
    const description = e.features[0].properties.description;
    while (Math.abs(e.lngLat.lng - coordinates[0]) > 180) {
      coordinates[0] += e.lngLat.lng > coordinates[0] ? 360 : -360;
    }
    popup.setLngLat(coordinates).setHTML(description).addTo(map);
  });

  map.on('mouseleave', 'places', () => {
    map.getCanvas().style.cursor = '';
    popup.remove();
  });
});


const activePara = document.getElementById('activePara');
const activeButton = document.getElementById('activeSensor');

// Add a click event listener to the button
activeButton.addEventListener('click', function() {
  // Change the text content of the paragraph when the button is clicked
  activePara.textContent = '22';
});
